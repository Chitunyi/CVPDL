# HW2 – Object Detection 

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
python preprocess.py \
  --raw_dir path/to/train \
  --out_dir ./dataset \
  --val_ratio 0.2 \
  --seed 42 \
  --make_rfs_balanced \
  --rfs_t 0.02 \
  --rfs_beta 0.5 \
  --rfs_bias_map 2:3.0 \
  --rfs_inst_boost 0.25 \
  --max_repeat 5 --link hard

## Training

python train.py \
  --data ./dataset/data_bal.yaml \
  --model yolov10n.yaml \
  --epochs 300 \
  --batch 2 \
  --imgsz 1920 \
  --device 0 \
  --mosaic 1.0 \
  --mixup 0.1 \
  --copy_paste 1.0 \
  --close_mosaic 30 \
  --cos_lr


## Prediction 
# Use tuned args:
python infer.py \
  --weights runs/hw2_from_scratch/yolov10_longtail/weights/best.pt \
  --test_dir path/to/test \
  --out_csv submission.csv \
  --img 2016

# The script preserves the Image_ID order from sample_csv and writes:
#   Image_ID,PredictionString
# PredictionString = "score x y w h class ..."

## Troubleshooting
- "Sample CSV must have 'Image_ID,PredictionString'": verify the header of sample_submission.csv.
- CUDA vs CPU device: infer.py auto-selects CUDA if available; override with --device "0" or --device "cpu".
