#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

CLASS_NAMES = ['car', 'hov', 'person', 'motorcycle']  

def xyxy_to_ltwh(xyxy):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True, help='Path to trained weights (.pt)')
    ap.add_argument('--test_dir', type=Path, required=True, help='Folder containing test images (img*.png)')
    ap.add_argument('--out_csv', type=Path, default=Path('submission.csv'))
    ap.add_argument('--img', type=int, default=2016)
    ap.add_argument('--conf', type=float, default=0.01)
    # ap.add_argument('--iou', type=float, default=0.50, help='NMS IoU threshold')
    # ap.add_argument('--max_det', type=int, default=40, help='Max detections per image')
    # ap.add_argument('--tta', action='store_true', help='Enable test-time augmentation')
    ap.add_argument('--device', type=str, default='0')
    args = ap.parse_args()

    model = YOLO(args.weights)

    test_imgs = sorted(list(args.test_dir.glob('img*.png')))
    assert test_imgs, f'No images found in {args.test_dir}'

    rows = []
    for idx, img_path in enumerate(test_imgs, start=1):  # Image_ID is 1-based index
        with Image.open(img_path) as im:
            W, H = im.size

        # res = model.predict(source=str(img_path), imgsz=args.img, conf=args.conf, device=args.device, verbose=False)[0]
        res = model.predict(
            source=str(img_path),
            imgsz=args.img,
            conf=args.conf,
            # iou=args.iou,
            # max_det=args.max_det,
            device=args.device,
            verbose=False,
            augment=args.tta  
        )[0]

        pred_str_parts = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                # xyxy in pixels, conf in [0,1], cls id
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                conf = float(b.conf[0].item())
                cls_id = int(b.cls[0].item())

                # clip to image bounds
                x1 = max(0.0, min(x1, W - 1))
                y1 = max(0.0, min(y1, H - 1))
                x2 = max(0.0, min(x2, W - 1))
                y2 = max(0.0, min(y2, H - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                left, top, width, height = xyxy_to_ltwh((x1, y1, x2, y2))

                # Round to integers as typically expected by competitions
                left, top, width, height = int(round(left)), int(round(top)), int(round(width)), int(round(height))

                pred_str_parts.extend([
                    f'{conf:.6f}', str(left), str(top), str(width), str(height), str(cls_id)
                ])

        prediction_string = ' '.join(pred_str_parts)
        rows.append((idx, prediction_string))

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_ID', 'PredictionString'])
        for r in rows:
            writer.writerow(r)

    print(f'[OK] Wrote submission CSV to: {args.out_csv}')
    print('Row example:', rows[0] if rows else '(no predictions)')

if __name__ == '__main__':
    main()
