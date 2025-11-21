#!/usr/bin/env python3

from __future__ import annotations
import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import shutil

CLASS_NAMES = ["car", "hov", "person", "motorcycle"]  # 0..3 

def _read_hw2_label(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Return list of (cls, x_left, y_top, w, h), floats in pixel units."""
    items = []
    for line in txt_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(',')
        if len(parts) != 5:
            raise ValueError(f'Bad label line in {txt_path}: {line}')
        c = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        items.append((c, x, y, w, h))
    return items

def _parse_bias_map(s: str) -> Dict[int, float]:
    bm = {}
    if not s:
        return bm
    for tok in s.split(','):
        k, v = tok.split(':')
        bm[int(k.strip())] = float(v.strip())
    return bm

def _to_yolo(items, img_w, img_h) -> List[Tuple[int, float, float, float, float]]:
    """Convert absolute (x_left,y_top,w,h) to YOLO (cls, cx/img_w, cy/img_h, w/img_w, h/img_h)."""
    out = []
    for c, x, y, w, h in items:
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        ww = w / img_w
        hh = h / img_h
        # Clamp
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        ww = min(max(ww, 0.0), 1.0)
        hh = min(max(hh, 0.0), 1.0)
        out.append((c, cx, cy, ww, hh))
    return out

def _imsize(img_path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(img_path) as im:
        w, h = im.size
    return w, h

def _write_yolo_label(yolo_items, out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open('w') as f:
        for c, cx, cy, ww, hh in yolo_items:
            f.write(f"{c} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")

def _hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)  
    except Exception:
        shutil.copy2(src, dst)

def build_base_dataset(raw_dir: Path, out_dir: Path, val_ratio: float, seed: int):
    random.seed(seed)
    out_images = out_dir / 'images'
    out_labels = out_dir / 'labels'
    (out_images / 'train').mkdir(parents=True, exist_ok=True)
    (out_images / 'val').mkdir(parents=True, exist_ok=True)
    (out_labels / 'train').mkdir(parents=True, exist_ok=True)
    (out_labels / 'val').mkdir(parents=True, exist_ok=True)

    # gather image ids
    imgs = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    assert imgs, f'No images found in {raw_dir}'
    ids = [p.stem for p in imgs]
    random.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_ratio)))
    val_ids = set(ids[:n_val])
    train_ids = [i for i in ids if i not in val_ids]

    total_objs = 0
    per_split_counts = Counter()

    for img_id in ids:
        img_path = raw_dir / f"{img_id}.png"
        if not img_path.exists():
            # try .jpg
            img_path = raw_dir / f"{img_id}.jpg"
        txt_path = raw_dir / f"{img_id}.txt"
        assert img_path.exists() and txt_path.exists(), f'Missing pair for {img_id}'

        labels = _read_hw2_label(txt_path)
        w, h = _imsize(img_path)
        yolo_items = _to_yolo(labels, w, h)
        total_objs += len(yolo_items)

        if img_id in val_ids:
            _hardlink_or_copy(img_path, out_images / 'val' / img_path.name)
            _write_yolo_label(yolo_items, out_labels / 'val' / f"{img_id}.txt")
            per_split_counts['val_imgs'] += 1
            per_split_counts['val_objs'] += len(yolo_items)
        else:
            _hardlink_or_copy(img_path, out_images / 'train' / img_path.name)
            _write_yolo_label(yolo_items, out_labels / 'train' / f"{img_id}.txt")
            per_split_counts['train_imgs'] += 1
            per_split_counts['train_objs'] += len(yolo_items)

    data_yaml = out_dir / 'data.yaml'
    data_yaml.write_text(f"""# auto-generated
        path: {out_dir.as_posix()}
        train: images/train
        val: images/val
        nc: {len(CLASS_NAMES)}
        names: {CLASS_NAMES}
        """)

    print(f"[OK] Base YOLO dataset @ {out_dir}")
    print(f"train_imgs={per_split_counts['train_imgs']}  val_imgs={per_split_counts['val_imgs']}  total={len(ids)}")
    print(f"train_objs={per_split_counts['train_objs']}  val_objs={per_split_counts['val_objs']}  total_objs={total_objs}")
    return {
        'train_dir': out_dir / 'images' / 'train',
        'train_lbl': out_dir / 'labels' / 'train',
        'val_dir': out_dir / 'images' / 'val',
        'val_lbl': out_dir / 'labels' / 'val',
        'data_yaml': data_yaml
    }

# ---------- RFS (LVIS-style) ----------
def _image_level_classes(label_txt: Path) -> Set[int]:
    s = set()
    if not label_txt.exists():
        return s
    for line in label_txt.read_text().strip().splitlines():
        if not line.strip():
            continue
        c = int(line.strip().split()[0])
        s.add(c)
    return s


def compute_rfs_repeat_factors(lbl_train_dir: Path, t: float = 0.001, cap: int = 5,
                               beta: float = 0.5, bias_map: Dict[int, float] | None = None,
                               inst_boost: float = 0.25) -> Dict[str, float]:
    from collections import defaultdict, Counter
    bias_map = bias_map or {}

    lf_paths = sorted(lbl_train_dir.glob('*.txt'))
    N = len(lf_paths)
    per_class_img_count = Counter()
    img_classes = {}
    img_class_inst = {}  

    def _image_level_counts(txt_path: Path):
        cls_set = set()
        counts = Counter()
        for line in txt_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            c = int(line.split()[0])
            cls_set.add(c)
            counts[c] += 1
        return cls_set, counts

    for lf in lf_paths:
        classes, counts = _image_level_counts(lf)
        img_classes[lf.stem] = classes
        img_class_inst[lf.stem] = counts
        for c in classes:
            per_class_img_count[c] += 1

    # image-level frequency f[c]
    f = {c: per_class_img_count[c] / max(N, 1) for c in range(len(CLASS_NAMES))}

    r_c = {}
    for c in range(len(CLASS_NAMES)):
        base = (t / max(f.get(c, 1e-12), 1e-12)) ** beta if f.get(c, 0) > 0 else float(cap)
        base = max(1.0, base)
        base *= bias_map.get(c, 1.0) 
        r_c[c] = min(base, cap)

    r_img = {}
    for stem, classes in img_classes.items():
        if not classes:
            r_img[stem] = 1.0
            continue
        best = 1.0
        counts = img_class_inst[stem]
        for c in classes:
            boost = 1.0 + inst_boost * math.log1p(counts.get(c, 0))
            best = max(best, r_c[c] * boost)
        r_img[stem] = float(min(best, cap))
    return r_img


def build_rfs_balanced_train(base_out: dict, out_dir: Path, t: float = 0.001, cap: int = 5, 
                             link_mode: str = "hard", beta: float = 0.5, 
                             bias_map_str: str = '', inst_boost: float = 0.25):
    bias_map = _parse_bias_map(bias_map_str)
    train_img_dir = base_out['train_dir']
    train_lbl_dir = base_out['train_lbl']

    img2r = compute_rfs_repeat_factors(train_lbl_dir, t=t, cap=cap, beta=beta, 
                                       bias_map=bias_map, inst_boost=inst_boost)

    out_img = out_dir / 'images' / 'train_bal'
    out_lbl = out_dir / 'labels' / 'train_bal'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    def link(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if link_mode == "hard":
            try:
                if dst.exists(): dst.unlink()
                os.link(src, dst)
            except Exception:
                shutil.copy2(src, dst)
        elif link_mode == "symlink":
            if dst.exists(): dst.unlink()
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

    # make duplicates
    import random
    total_out = 0
    for img_path in sorted(train_img_dir.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        stem = img_path.stem
        lbl_path = train_lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            # keep in base dataset as-is
            k = 1
            frac = 0.0
        else:
            r = img2r.get(stem, 1.0)
            k = int(math.floor(r))
            frac = r - k
            if random.random() < frac:
                k += 1
            k = max(1, min(k, cap))
        # produce k copies (including the original name as _rep0 for clarity)
        for rep in range(k):
            suffix = f"_rep{rep}"
            dst_img = out_img / f"{stem}{suffix}{img_path.suffix.lower()}"
            dst_lbl = out_lbl / f"{stem}{suffix}.txt"
            link(img_path, dst_img)
            if lbl_path.exists():
                link(lbl_path, dst_lbl)
            total_out += 1

    data_bal_yaml = out_dir / 'data_bal.yaml'
    data_bal_yaml.write_text(f"""# auto-generated (RFS balanced)
        path: {out_dir.as_posix()}
        train: images/train_bal
        val: images/val
        nc: {len(CLASS_NAMES)}
        names: {CLASS_NAMES}
        """)
    print(f"[OK] Built RFS-balanced train split @ {out_img}")
    print(f"     Repeated images (after rounding/cap) = {total_out}")


    from collections import Counter
    before = Counter()
    after = Counter()
    for lf in sorted(train_lbl_dir.glob('*.txt')):
        classes = _image_level_classes(lf)
        for c in classes:
            before[c] += 1
        r = int(round(img2r[lf.stem]))
        for c in classes:
            after[c] += r
    print('[RFS] image-level counts by class (before -> after):')
    for cid, name in enumerate(CLASS_NAMES):
        print(f'{cid}:{name:>10s}  {before[cid]:6d} -> {after[cid]:6d}')

    return data_bal_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', type=Path, required=True)
    ap.add_argument('--out_dir', type=Path, required=True)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=1337)
    # long-tail options
    ap.add_argument('--make_rfs_balanced', action='store_true', help='create train_bal via LVIS-style RFS')
    ap.add_argument('--rfs_t', type=float, default=0.001, help='RFS threshold t')
    ap.add_argument('--rfs_beta', type=float, default=0.5, help='RFS exponent beta (0.5 = sqrt)')
    ap.add_argument('--rfs_bias_map', type=str, default='2:2.0', 
                    help='per-class bias like "2:2.0,3:1.2" (class_id:multiplier)')
    ap.add_argument('--rfs_inst_boost', type=float, default=0.25, 
                    help='per-image instance-count boost for rare classes (0 to disable)')
    ap.add_argument('--max_repeat', type=int, default=5, help='cap on repeats per image')
    ap.add_argument('--link', type=str, default='hard', choices=['hard', 'copy', 'symlink'], help='how to duplicate files')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_out = build_base_dataset(args.raw_dir, args.out_dir, args.val_ratio, args.seed)

    if args.make_rfs_balanced:
        data_bal_yaml = build_rfs_balanced_train(base_out, args.out_dir, t=args.rfs_t, cap=args.max_repeat, link_mode=args.link)
        print(f"[OK] data_bal.yaml -> {data_bal_yaml}")
    else:
        print(f"[OK] data.yaml -> {base_out['data_yaml']}")

if __name__ == '__main__':
    main()
