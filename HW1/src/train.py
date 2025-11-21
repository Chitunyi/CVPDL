#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import torch
import yaml

from PIL import Image

# Ultralytics (YOLOv8/YOLOv11) — pip install ultralytics
from ultralytics import YOLO


def _glob_images(images_dir: Path) -> Dict[str, Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    files = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    key_map = {}
    for p in files:
        key_map[p.stem] = p
    return key_map


def _find_image_for_id(img_id: int, images_map: Dict[str, Path]) -> Optional[Path]:
    s1 = str(img_id)
    s2 = f"{img_id:08d}"
    if s1 in images_map:
        return images_map[s1]
    if s2 in images_map:
        return images_map[s2]
    # fallback: startswith
    for stem, path in images_map.items():
        if stem == s1 or stem == s2 or stem.startswith(s1):
            return path
    return None


def _read_gt(gt_txt: Path) -> Dict[int, List[Tuple[float, float, float, float, int]]]:
    boxes = defaultdict(list)
    with gt_txt.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [t for t in line.replace(",", " ").split() if t]
            if len(parts) < 5:
                raise ValueError(f"[gt] line {line_num}: expect at least 5 fields, got {len(parts)} -> {parts}")
            img_id = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            cls = int(parts[5]) if len(parts) >= 6 else 0
            boxes[img_id].append((x, y, w, h, cls))
    return boxes


def _to_yolo_label(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    (x_min, y_min, w, h) -> YOLO normalized (x_center, y_center, w, h)
    """
    x_c = x + w / 2.0
    y_c = y + h / 2.0
    return x_c / img_w, y_c / img_h, w / img_w, h / img_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=Path, required=True, help="Folder that contains all training images")
    ap.add_argument("--gt_txt", type=Path, required=True, help="Path to gt.txt")
    ap.add_argument("--workdir", type=Path, default=Path("./workdir"), help="Where to write dataset & runs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--val_ratio", type=float, default=0.0, help="Fraction of images for validation split")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    ap.add_argument("--use_aug", action="store_true", help="Write a custom hyp YAML with moderate augmentation")
    ap.add_argument("--hyp_out", type=Path, default=Path("./workdir/hyp_custom.yaml"), help="Path to write hyp YAML if --use_aug")
    ap.add_argument("--lr0", type=float, default=None, help="Initial learning rate (Ultralytics arg lr0)")
    ap.add_argument("--lrf", type=float, default=None, help="Final LR ratio (0~1), Ultralytics arg lrf")
    ap.add_argument("--device", type=str, default=None, help="Set to '0' for first GPU, or 'cpu'. Leave empty to auto-select.")
    ap.add_argument("--backbone_weights", type=Path, default=None,
                    help="(Optional) Path to self-supervised ImageNet weights for BACKBONE ONLY (advanced use).")
    ap.add_argument("--close_mosaic_ratio", type=float, default=0.3,
                help="Disable mosaic/mixup in the last ratio of epochs")
    args = ap.parse_args()

    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
    else:
        if args.device.lower() in {"auto", "cuda", "gpu"}:
            args.device = "0" if torch.cuda.is_available() else "cpu"

    print(f"[device] using device={args.device}  (cuda_available={torch.cuda.is_available()})")


    args.workdir.mkdir(parents=True, exist_ok=True)

    images_map = _glob_images(args.images_dir)
    assert images_map, f"No images found under {args.images_dir}"
    gt = _read_gt(args.gt_txt)

    ds_root = args.workdir / "dataset"
    # img_out = ds_root / "images" / "train"
    # lbl_out = ds_root / "labels" / "train"
    # img_out.mkdir(parents=True, exist_ok=True)
    # lbl_out.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(args.seed)
    all_imgs = sorted([p for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    n_total = len(all_imgs)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    val_imgs = set(random.sample(all_imgs, n_val))
    print(f"[split] total={n_total}  val={len(val_imgs)}  train={n_total - len(val_imgs)} (seed={args.seed})")
    img_out_tr = ds_root / "images" / "train"
    img_out_va = ds_root / "images" / "val"
    lbl_out_tr = ds_root / "labels" / "train"
    lbl_out_va = ds_root / "labels" / "val"
    img_out_tr.mkdir(parents=True, exist_ok=True)
    img_out_va.mkdir(parents=True, exist_ok=True)
    lbl_out_tr.mkdir(parents=True, exist_ok=True)
    lbl_out_va.mkdir(parents=True, exist_ok=True)


    used = 0
    for img_id, bxs in gt.items():
        img_path = _find_image_for_id(img_id, images_map)
        if img_path is None:
            print(f"[warn] image for id={img_id} not found, skip")
            continue

        # dst_img = img_out / img_path.name
        is_val = img_path in val_imgs
        # is_val = (img_path.name in val_names)
        dst_img = (img_out_va if is_val else img_out_tr) / img_path.name
        if not dst_img.exists():
            try:
                os.symlink(img_path.resolve(), dst_img)
            except OSError:
                shutil.copyfile(img_path, dst_img)

        with Image.open(img_path) as im:
            w, h = im.size

        # dst_lbl = lbl_out / (img_path.stem + ".txt")
        is_val = img_path in val_imgs
        # is_val = (img_path.name in val_names)
        dst_lbl = (lbl_out_va if is_val else lbl_out_tr) / (img_path.stem + ".txt")
        with dst_lbl.open("w", encoding="utf-8") as f:
            for (x, y, bw, bh, cls) in bxs:
                xc, yc, nw, nh = _to_yolo_label(x, y, bw, bh, w, h)
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                nw = min(max(nw, 0.0), 1.0)
                nh = min(max(nh, 0.0), 1.0)
                f.write(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
        used += 1

    assert used > 0, "No labels written (check gt.txt format and image_id mapping)."
    print(f"[ok] wrote labels for {used} images at {lbl_out_tr}")

    data_yaml = args.workdir / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        f.write(
            f"path: {ds_root.as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names: [\"object\"]\n"
        )
    print(f"[ok] wrote {data_yaml}")

    model = YOLO("yolo11n.yaml") 

    # Data augmentation hyp
    hyp_path = None
    if args.use_aug:
        hyp_path = args.hyp_out
        hyp_path.parent.mkdir(parents=True, exist_ok=True)
        hyp_text = """# stronger augmentation 
            lr0: 0.005          
            lrf: 0.05         
            momentum: 0.937
            weight_decay: 0.0005
            warmup_epochs: 3.0
            warmup_momentum: 0.8
            warmup_bias_lr: 0.1
            box: 7.5
            cls: 0.5
            dfl: 1.5
            hsv_h: 0.015
            hsv_s: 0.7
            hsv_v: 0.4
            degrees: 5.0
            translate: 0.06
            scale: 0.65
            shear: 0.0
            perspective: 0.0
            flipud: 0.0
            fliplr: 0.5
            mosaic: 0.5        
            mixup: 0.1        
            copy_paste: 0.0
            """

        hyp_path.write_text(hyp_text, encoding="utf-8")
        print(f"[aug] wrote hyp yaml -> {hyp_path}")

    extra_kwargs = {}
    if args.use_aug:
        with open(hyp_path, "r", encoding="utf-8") as f:
            aug_cfg = yaml.safe_load(f) or {}
        allowed = {
            "lr0","lrf","momentum","weight_decay","warmup_epochs","warmup_momentum","warmup_bias_lr",
            "box","cls","dfl",
            "hsv_h","hsv_s","hsv_v","degrees","translate","scale","shear","perspective",
            "flipud","fliplr","mosaic","mixup","copy_paste"
        }
        extra_kwargs.update({k:v for k,v in aug_cfg.items() if k in allowed})

    if args.lr0 is not None:
        extra_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        extra_kwargs["lrf"] = args.lrf

    close_epochs = max(1, int(args.epochs * args.close_mosaic_ratio))
    print(f"[aug] close_mosaic set to last {close_epochs} epochs "
        f"({args.close_mosaic_ratio*100:.0f}% of {args.epochs} epochs)")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        pretrained=False,
        project=str(args.workdir),
        name="train",
        exist_ok=True,
        close_mosaic=close_epochs,
        **extra_kwargs,  
    )


    print("[done] training finished. Check weights under workdir/train/weights/")

if __name__ == "__main__":
    sys.exit(main())
