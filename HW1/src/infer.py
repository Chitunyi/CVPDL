#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np


def _glob_images(images_dir: Path) -> Dict[str, Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    files = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    return {p.stem: p for p in files}


def _resolve_id_to_path(img_id: int, images_map: Dict[str, Path]) -> Optional[Path]:
    s1 = str(img_id)
    s2 = f"{img_id:08d}"
    if s1 in images_map:
        return images_map[s1]
    if s2 in images_map:
        return images_map[s2]
    # fallback: any stem starting with the id (common in comps)
    for stem, p in images_map.items():
        if stem == s1 or stem == s2 or stem.startswith(s1):
            return p
    return None


def _boxes_to_prediction_string(
    xyxy: np.ndarray, conf: np.ndarray, img_w: int, img_h: int, classes: np.ndarray
) -> str:
    """
    Convert (x1,y1,x2,y2) -> "score x_min y_min width height cls" 
    """
    if xyxy.size == 0:
        return ""
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    x = xyxy[:, 0]
    y = xyxy[:, 1]

    # clamp to image bounds
    x = np.clip(x, 0, img_w - 1e-3)
    y = np.clip(y, 0, img_h - 1e-3)
    w = np.clip(w, 0, img_w - x)
    h = np.clip(h, 0, img_h - y)

    parts: List[str] = []
    for i in range(len(conf)):
        sc = float(conf[i])
        xi = float(x[i]); yi = float(y[i]); wi = float(w[i]); hi = float(h[i])
        ci = int(classes[i]) if classes is not None else 0
        parts.append(f"{sc:.6f} {xi:.2f} {yi:.2f} {wi:.2f} {hi:.2f} {ci}")
    return " ".join(parts)

def _grid_search_conf_iou(model, data_yaml, imgsz=640, device="cpu"):
    conf_list = [0.35, 0.40, 0.45, 0.50, 0.55]
    iou_list  = [0.55, 0.60, 0.65, 0.70]

    best = {"map50": -1.0, "conf": None, "iou": None}
    for conf in conf_list:
        for iou in iou_list:
            r = model.val(data=data_yaml, imgsz=imgsz, conf=conf, iou=iou, device=device, plots=False, verbose=False)
            try:
                map50 = float(getattr(r.box, "map50", None) or r.results_dict.get("metrics/mAP50(B)", -1.0))
            except Exception:
                map50 = -1.0
            if map50 > best["map50"]:
                best.update({"map50": map50, "conf": conf, "iou": iou})
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="Path to trained weights, e.g., workdir/train/weights/best.pt")
    ap.add_argument("--images_dir", type=Path, required=True, help="Folder of test images")
    ap.add_argument("--sample_csv", type=Path, required=True, help="Sample submission CSV to copy Image_ID order")
    ap.add_argument("--out_csv", type=Path, default=Path("./submission.csv"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--max_det", type=int, default=25, help="Maximum detections per image")
    ap.add_argument("--tta", action="store_true", help="Enable test-time augmentation (Ultralytics)")
    ap.add_argument("--agnostic_nms", action="store_true", help="Use class-agnostic NMS")
    ap.add_argument("--val_yaml", type=str, default=None, help="Path to data.yaml that contains val split")
    ap.add_argument("--tune_conf", action="store_true", help="Grid search conf/iou on val split to pick best")
    ap.add_argument("--save_best_args", type=str, default=None, help="If set, save the picked best conf/iou to this JSON file")
    ap.add_argument("--device", type=str, default=None, help="Set '0' for first GPU or 'cpu'. Leave empty to auto-select.")
    args = ap.parse_args()

    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
    else:
        dv = str(args.device).lower()
        if dv in {"auto", "cuda", "gpu"}:
            args.device = "0" if torch.cuda.is_available() else "cpu"

    print(f"[device] using device={args.device} (cuda_available={torch.cuda.is_available()})")

    sample = pd.read_csv(args.sample_csv)
    assert list(sample.columns)[:2] == ["Image_ID", "PredictionString"], "Sample CSV must have 'Image_ID,PredictionString' columns"

    images_map = _glob_images(args.images_dir)
    if not images_map:
        raise FileNotFoundError(f"No images found under: {args.images_dir}")

    model = YOLO(str(args.weights))

    preds: List[str] = []
    for img_id in sample["Image_ID"].tolist():
        p = _resolve_id_to_path(int(img_id), images_map)
        if p is None:
            preds.append("")
            continue
        
        results = model.predict(
            source=str(p),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            augment=args.tta,
            agnostic_nms=args.agnostic_nms,
            verbose=False
        )
        r = results[0]

        with Image.open(p) as im:
            img_w, img_h = im.size

        if r.boxes is None or r.boxes.xyxy.numel() == 0:
            preds.append("")
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else np.zeros(len(conf), dtype=int)

        ps = _boxes_to_prediction_string(xyxy, conf, img_w, img_h, cls)
        preds.append(ps)

    out_df = pd.DataFrame({"Image_ID": sample["Image_ID"].astype(int), "PredictionString": preds})
    out_df.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote: {args.out_csv}  (rows={len(out_df)})")

if __name__ == "__main__":
    main()
