# Assignment 07 — Transfer Learning for Grapevine Leaf Classification

## Problem Statement

Implement transfer learning for image classification using a Grapevine Leaves dataset and compare the performance of different pretrained deep learning architectures.

## Objective

- Apply transfer learning to a small image-classification dataset.
- Compare pretrained CNN architectures for grapevine leaf classification.
- Train and evaluate AlexNet, VGG16, ResNet50, and EfficientNetB0.
- Analyze performance using validation accuracy, test accuracy, loss curves, classification reports, and confusion matrices.

## Dataset

**Dataset:** Grapevine Leaves Image Dataset

The experiment uses 5 grapevine leaf classes:

- Ak
- Ala_Idris
- Buzgulu
- Dimnit
- Nazli

The dataset was split into:

| Split | Images |
|---|---:|
| Training | 350 |
| Validation | 75 |
| Testing | 75 |
| Total | 500 |

The dataset itself is not included in this repository because image datasets can be large. The provided `split_dataset.py` script can be used to create the train/validation/test structure from the original dataset.

## Methodology / Workflow

1. Load the Grapevine Leaves image dataset.
2. Split the original images into training, validation, and test sets using a fixed seed.
3. Resize images to 224 × 224.
4. Apply training-time augmentation using horizontal flipping, rotation, and color jitter.
5. Normalize images using ImageNet normalization.
6. Load pretrained CNN architectures.
7. Freeze most pretrained feature-extraction layers and fine-tune the deeper layers.
8. Replace the original classifier with a 5-class classifier.
9. Train each model for 15 epochs using AdamW.
10. Evaluate the trained models on the held-out test set.
11. Generate classification reports and confusion matrices.
12. Compare the four architectures using accuracy, loss, and training time.

## Model / Technique

The experiment compares four pretrained image-classification architectures:

- **AlexNet** — baseline pretrained CNN architecture.
- **VGG16** — deeper convolutional architecture with stacked convolution layers.
- **ResNet50** — residual network using skip connections for deeper feature learning.
- **EfficientNetB0** — efficient architecture designed for a strong accuracy-to-computation trade-off.

### Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 15 |
| Optimizer | AdamW |
| Classifier Learning Rate | 0.001 |
| Fine-tuned Feature Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Random Seed | 42 |
| Number of Classes | 5 |

## Results

### Model Comparison

| Model | Best Epoch | Best Validation Accuracy | Test Accuracy | Test Loss | Training Time |
|---|---:|---:|---:|---:|---:|
| AlexNet | 11 | 78.67% | 78.67% | 0.9192 | 126.30 s |
| VGG16 | 10 | 90.67% | 81.33% | 1.2675 | 909.15 s |
| ResNet50 | 6 | **94.67%** | **89.33%** | 0.4067 | 644.00 s |
| EfficientNetB0 | 14 | 89.33% | 88.00% | **0.3958** | **398.06 s** |

### Interpretation

**ResNet50 achieved the highest test accuracy at 89.33%** and also recorded the highest validation accuracy at 94.67%.

**EfficientNetB0 was a close second**, reaching 88.00% test accuracy while requiring substantially less training time than ResNet50. It also produced the lowest test loss among the four models at 0.3958.

VGG16 achieved 81.33% test accuracy, while AlexNet achieved 78.67%. The results show that model architecture and transfer-learning capacity have a noticeable effect when working with a relatively small image dataset.

### Confusion Matrices and Training Curves

The `results/` directory contains the saved accuracy curves, loss curves, and test-set confusion matrices for all four models.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

## How to Run

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place the original dataset at:

```text
Dataset/Grapevine_Leaves_Image_Dataset/
```

Then run:

```bash
python split_dataset.py
```

This creates:

```text
Dataset/Split/
├── train/
├── val/
└── test/
```

### 4. Train a model

Run any of the following:

```bash
python train_alexnet.py
python train_vgg16.py
python train_resnet50.py
python train_efficientnetb0.py
```

The scripts save evaluation results and graphs inside `results/`.

> The repository contains the saved results from the completed experiments, so the models do not need to be retrained just to inspect the reported outcomes.
