# Medical Image Classification using CNNs

Brain tumor classification from MRI scans using custom CNNs, transfer learning, and explainable AI (Grad-CAM).

## Overview

This project builds and compares deep learning models for detecting the presence of brain tumors in MRI images. It includes:

- **Custom CNN architectures** trained from scratch
- **Transfer learning models** (MobileNetV2, SqueezeNet, ResNet18)
- **Explainable AI** via Grad-CAM to visualize model decision regions

## Dataset

- **Source:** [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019.html)
- **Task:** Binary classification — Tumor (Yes) vs. No Tumor (No)

### Expected structure

```
data/
└── raw/
    └── BraTS2019 dataset/
        ├── train/
        ├── valid/
        └── test/
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Data preparation

1. Place the BraTS2019 dataset inside `data/raw/`.
2. Validate the dataset:
   ```bash
   python scripts/data_validation.py
   ```
3. Preprocess the data:
   ```bash
   python scripts/preprocessing.py
   ```

### Train custom CNN

```bash
python -m training.train_custom
```

Saved models: `results/custom_models/`

### Train pretrained models

Trains MobileNetV2, SqueezeNet, and ResNet18:

```bash
python -m training.train_pretrained
```

Saved models: `results/pretrained_models/`

### Evaluation

```bash
python -m training.evaluate_custom       # Custom CNN
python -m training.evaluate_pretrained   # Pretrained models
```

### Explainable AI (Grad-CAM)

```bash
python -m xai.gradcam_custom       # Custom CNN
python -m xai.gradcam_pretrained   # ResNet18
```

Outputs: `results/xai/`

## Project Structure

```
├── data/
│   └── raw/
├── scripts/
│   ├── data_validation.py
│   └── preprocessing.py
├── training/
│   ├── train_custom.py
│   ├── train_pretrained.py
│   ├── evaluate_custom.py
│   └── evaluate_pretrained.py
├── xai/
│   ├── gradcam_custom.py
│   └── gradcam_pretrained.py
├── results/
│   ├── custom_models/
│   ├── pretrained_models/
│   └── xai/
└── requirements.txt
```

## Contributing

Commit convention: `TaskX.Y: description`

```bash
git add .
git commit -m "TaskX.Y: description"
git push
```
