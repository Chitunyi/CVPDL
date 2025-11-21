from ultralytics import YOLO
from itertools import product
import json, os

best = "./work_yolo11/train/weights/best.pt"
data_yaml = "./work_yolo11/data.yaml"   

model = YOLO(best)

# === SPACE ===
imgsz_list = [960]
conf_list  = [0.30, 0.35, 0.40, 0.45, 0.50]
iou_list   = [0.50, 0.55, 0.60, 0.65]
maxdet_list= [30, 35, 40, 45]
tta_list   = [False, True]

best_cfg = None
best_metric = -1.0

for imgsz, conf, iou, max_det, tta in product(imgsz_list, conf_list, iou_list, maxdet_list, tta_list):
    res = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        augment=tta,     
        plots=False,
        save=False,
        batch=16         
    )
    # mAP50 = float(getattr(res.metrics.box, "map50", 0.0))
    mAP50 = float(getattr(res.box, "map50", 0.0))
    mAP    = float(getattr(res.box, "map", 0.0))   
    
    if mAP50 > best_metric:
        best_metric = mAP50
        best_cfg = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, tta=tta)
        print("[*] NEW BEST", best_cfg, "mAP50=", round(best_metric, 5))

os.makedirs(workdir, exist_ok=True)
with open(os.path.join(workdir, "best_infer_cfg.json"), "w") as f:
    json.dump(best_cfg, f, indent=2)
print(f"[tune] saved best infer cfg -> {os.path.join(workdir, 'best_infer_cfg.json')}")

print("\n=== Best on val ===")
print(best_cfg, "mAP50=", round(best_metric, 5))
