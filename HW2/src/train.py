#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from ultralytics import YOLO

def get_args():
    ap = argparse.ArgumentParser()
    # dataset / model
    ap.add_argument('--data', type=str, required=True, help='path to YOLO data.yaml')
    ap.add_argument('--model', type=str, default='yolov10n.yaml', help='YOLOv10 model yaml (n/s/m/l/x)')
    # training core
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--imgsz', type=int, default=1920)
    ap.add_argument('--device', type=str, default='0')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8)
    # optimizer / lr
    ap.add_argument('--optimizer', type=str, default='SGD', choices=['SGD','Adam','AdamW','RMSProp'])
    ap.add_argument('--lr0', type=float, default=0.1)
    ap.add_argument('--momentum', type=float, default=0.937)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--cos_lr', action='store_true', help='use cosine LR schedule')
    ap.add_argument('--patience', type=int, default=100)
    # augmentation knobs
    ap.add_argument('--mosaic', type=float, default=1.0)
    ap.add_argument('--mixup', type=float, default=0.5)
    ap.add_argument('--copy_paste', type=float, default=0.0, help='0.0~1.0; good for long-tail rare classes')
    ap.add_argument('--degrees', type=float, default=0.0)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--translate', type=float, default=0.1)
    ap.add_argument('--hsv_h', type=float, default=0.015)
    ap.add_argument('--hsv_s', type=float, default=0.7)
    ap.add_argument('--hsv_v', type=float, default=0.4)
    ap.add_argument('--close_mosaic', type=int, default=10, help='disable mosaic for last K epochs')
    # misc
    ap.add_argument('--project', type=str, default='runs/hw2_from_scratch')
    ap.add_argument('--name', type=str, default='yolov10_longtail')
    ap.add_argument('--resume', action='store_true')
    return ap.parse_args()

def main():
    args = get_args()

    # Build model from yaml — ensures RANDOM INIT (no pretrained weights)
    model = YOLO(args.model)  

    # Ultralytics model.train kwargs
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        # optimization
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        patience=args.patience,
        cos_lr=args.cos_lr,
        # augmentation
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        degrees=args.degrees,
        scale=args.scale,
        translate=args.translate,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        close_mosaic=args.close_mosaic,
        # bookkeeping
        project=args.project,
        name=args.name,
        exist_ok=True,
        # ABSOLUTELY NO PRETRAINED
        pretrained=False,
        resume=args.resume,
    )

    print('[INFO] Training with args:')
    for k, v in train_kwargs.items():
        print(f'  - {k}: {v}')
    model.train(**train_kwargs)
    print(f'[OK] Training complete. See {Path(args.project)/args.name}')

if __name__ == '__main__':
    main()
