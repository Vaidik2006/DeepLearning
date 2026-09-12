# Assignment 06 — CNN for Tomato Disease Classification

## Problem Statement

Design and implement a Convolutional Neural Network (CNN) for image classification using the Tomato or Soybean disease dataset.

## Objective

Develop a CNN model for multi-class tomato leaf disease classification and evaluate its performance using training and validation accuracy and loss.

## Dataset

**Dataset:** Tomato disease image dataset

The practical uses separate `train` and `val` directories containing 10 tomato disease/health classes. Images are resized to **128 × 128 × 3** and normalized using `Rescaling(1./255)`.

## Methodology / Workflow

1. Load images from class-based train and validation directories.
2. Resize images to 128 × 128 pixels.
3. Normalize pixel values.
4. Visualize sample images.
5. Build a three-block CNN.
6. Compile with Adam and sparse categorical crossentropy.
7. Train for 15 epochs.
8. Evaluate validation loss and accuracy.
9. Plot training vs validation accuracy.
10. Plot training vs validation loss.
11. Save the trained model in `.keras` format.

## Model / Technique

```text
Input (128×128×3)
→ Rescaling
→ Conv2D(32) → MaxPooling2D
→ Conv2D(64) → MaxPooling2D
→ Conv2D(128) → MaxPooling2D
→ Flatten
→ Dense(128, ReLU)
→ Dropout(0.5)
→ Dense(10, Softmax)
```

**Training configuration:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 15
- Image size: 128 × 128
- Dropout: 0.5
- Output classes: 10

## Results

The submitted practical recorded:

| Metric | Result |
|---|---:|
| Validation Loss | 0.5064 |
| Validation Accuracy | 85.50% |

The practical sheet also includes training-vs-validation accuracy and loss plots. Training accuracy rises steadily, while validation accuracy fluctuates during later epochs. Validation loss also becomes more variable toward the end.

## Technologies

- Python
- TensorFlow / Keras
- Matplotlib
- Google Colab
- GPU acceleration

## How to Run

### Google Colab

1. Upload `Assignment_06_CNN_Tomato_Disease_Classification.ipynb`.
2. Upload/extract the tomato dataset with `train` and `val` folders.
3. Ensure the 10 class folders are inside both directories.
4. Run the notebook from top to bottom.
5. Review the model summary, validation metrics and plots.

Update `TRAIN_DIR` and `VAL_DIR` if the dataset is stored elsewhere.

### Repository Contents

```text
Assignment6/
├── Assignment_06_CNN_Tomato_Disease_Classification.ipynb
├── Problem_Statement.pdf
├── README.md
└── requirements.txt
```