# Cloud Segmentation — Understanding Clouds from Satellite Images

DINOv3 SAT ViT-L/16 + FPN decoder for cloud segmentation.

## 1. Environment Setup

```bash
conda create -n cloud_seg python=3.10 -y
conda activate cloud_seg

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install albumentations pandas scikit-learn iterative-stratification tensorboard tqdm
pip install git+https://github.com/huggingface/transformers.git
```

## 2. HuggingFace Login

The DINOv3 SAT model requires authenticated access:

```bash
pip install huggingface_hub
huggingface-cli login
# Enter HuggingFace Access Token 
```

## 3. Data Download

Download the competition data and place it under `data/`:

```bash
pip install kaggle

cd /path/to/cloud_seg
mkdir -p data
cd data
kaggle competitions download -c understanding_cloud_organization
unzip understanding_cloud_organization.zip
rm understanding_cloud_organization.zip
cd ..
```

Expected directory structure:
```
data/
├── train.csv
├── sample_submission.csv
├── train_images/    # 5546 images, 1400x2100
└── test_images/     # 3698 images, 1400x2100
```

## 4. Data Preparation

Generate fold-split CSV:

```bash
python prepare_data.py
```

## 5. Training

```bash
# Single fold
python train.py --fold 0

# All 5 folds
for i in 0 1 2 3 4; do python train.py --fold $i; done
```

## 6. Threshold Search (5 fold)

Find optimal checkpoint combination and thresholds via OOF grid search:

```bash
# Step 1: Collect OOF predictions (requires GPU)
python threshold_search_5fold.py --collect

# Step 2: Grid search over 3^5 checkpoint combos x threshold pairs (CPU only)
python threshold_search_5fold.py --search
```

This outputs the best checkpoint combination, cls_threshold, and pixel_threshold.

## 7. Inference

```bash
# 5-fold ensemble inference (use thresholds from grid search)
python inference.py --ensemble --ckpts 1,1,1,1,1 --cls-threshold 0.6 --pixel-threshold 0.4
```
