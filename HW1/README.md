# HW1 – Object Detection 

## Environment
- Python ≥ 3.10  (required by course spec)
- Create venv and install:
  python -m venv .venv
  # macOS/Linux
  source .venv/bin/activate
  # Windows
  # .venv\Scripts\activate
  pip install -r requirements.txt

## Data preparation
- Train images folder: ./path/to/train/img
- Ground truth file:   ./path/to/train/gt.txt
  Each line in gt.txt: <Image_ID> <bb_left> <bb_top> <bb_width> <bb_height>
- Test images folder:  ./path/to/test/img

## Training

python train.py \
  --images_dir ./path/to/train/img \
  --gt_txt    ./path/to/train/gt.txt \
  --workdir   ./work_yolo11 \
  --epochs    120 \
  --batch     16 \
  --imgsz     960 \
  --val_ratio 0.2 \
  --seed      42 \
  --use_aug \
  --close_mosaic_ratio 0.1

# Notes
# - Set --val_ratio to create a validation split for tuning.
# - --use_aug writes <workdir>/hyp_custom.yaml and applies it to training.
# - --close_mosaic_ratio disables mosaic/mixup for the last ratio of epochs.

## Tune inference on validation
# Grid-search imgsz/conf/iou/max_det/tta using the trained model and <workdir>/data.yaml
python tune_para.py
# Output:
#   <workdir>/best_infer_cfg.json
#   Example: {"imgsz":960,"conf":a,"iou":b,"max_det":c,"tta":true}


## Prediction 
# Use tuned args:
python infer.py \
  --weights   ./work_yolo11/train/weights/best.pt \
  --images_dir ./path/to/test/img \
  --sample_csv ./path/to/sample_submission.csv \
  --out_csv    ./submission.csv \
  --imgsz 960 \
  --conf  a  \
  --iou   b  \
  --max_det c \
  --agnostic_nms \
  --tta

# The script preserves the Image_ID order from sample_csv and writes:
#   Image_ID,PredictionString
# PredictionString = "score x y w h class ..." (denormalized, class=0 for pigs)

## Troubleshooting
- "Sample CSV must have 'Image_ID,PredictionString'": verify the header of sample_submission.csv.
- CUDA vs CPU device: infer.py auto-selects CUDA if available; override with --device "0" or --device "cpu".
