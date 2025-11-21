# HW3 – Image Generation

## Environment
- Python ≥ 3.10  (required by course spec)
- Create venv and install:
  python -m venv .venv
  # macOS/Linux
  source .venv/bin/activate
  # Windows
  # .venv\Scripts\activate
  pip install -r requirements.txt

## Download Dataset
# The MNIST RGB image dataset used in this assignment is provided by the TA.
pip install gdown
gdown "https://drive.google.com/uc?id=1xVCJD6M6sE-tZJYxenLzvuEkSiYXig_F" -O mnist_rgb.zip
unzip mnist_rgb.zip -d /path/to/MNIST

## FID Implementation
pip install pytorch-fid

## Training
python train.py --data_dir /path/to/MNIST

## Prediction 
python infer.py --ckpt ./img_gen/ddpm_mnist.pth

## Score
python -m pytorch_fid ./generated/ /path/to/MNIST

