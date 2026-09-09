# Assignment 01 — Data Preprocessing, Normalization & Visualization

> **Deep Learning Laboratory | Practical 01**

## 📌 Problem Statement

Perform **data preprocessing, normalization, train-test splitting, and data visualization** on a sample dataset.

## 🎯 Objective

The objective of this practical is to understand the basic data preparation workflow required before training a deep learning model. The experiment demonstrates:

- Loading a sample dataset using TensorFlow/Keras
- Understanding the structure of image data
- Working with training and testing datasets
- Normalizing pixel values
- Visualizing sample data
- Training a basic neural network on the prepared data
- Evaluating the model on unseen test data

## 📊 Dataset

### MNIST Handwritten Digit Dataset

The experiment uses the **MNIST dataset**, which contains grayscale images of handwritten digits from **0 to 9**.

| Property | Value |
|---|---:|
| Training images | 60,000 |
| Testing images | 10,000 |
| Image size | 28 × 28 pixels |
| Number of classes | 10 |
| Pixel range before normalization | 0–255 |
| Pixel range after normalization | 0–1 |

The Keras dataset loader provides the training and testing sets separately, so the practical verifies this predefined train-test split rather than creating a second split.

## 🔄 Workflow

```text
MNIST Dataset
      ↓
Load Training & Testing Data
      ↓
Inspect Dataset Dimensions
      ↓
Normalize Pixel Values
      ↓
Visualize Sample Image
      ↓
Build Neural Network
      ↓
Train Model
      ↓
Evaluate on Test Data
      ↓
Visualize Training Performance
      ↓
Save Model
```

## 🧠 Model Used

A simple fully connected neural network is used to demonstrate the complete preprocessing-to-training workflow.

```text
Input: 28 × 28 image
        ↓
Flatten
        ↓
Dense Layer — 128 neurons, ReLU
        ↓
Dense Layer — 10 neurons, Softmax
        ↓
Digit Classification (0–9)
```

## 📈 Results

The original experiment trained the model for **3 epochs** and achieved approximately:

**Test Accuracy: 97.02%**

Training accuracy progressed from approximately **92.54% → 96.60% → 97.61%** across the three epochs.

> Results are based on the submitted experiment. Re-running the notebook may produce slightly different values depending on the TensorFlow/Keras environment and execution conditions.

## 🛠️ Technologies

- Python
- TensorFlow
- Keras
- Matplotlib
- Google Colab / Jupyter Notebook

## ▶️ How to Run

### Google Colab

1. Open `Assignment_01_Data_Preprocessing_MNIST.ipynb` in Google Colab.
2. Select **Runtime → Run all**.
3. The MNIST dataset will be loaded automatically through Keras.
4. Review the preprocessing, visualization, training, and evaluation outputs.

### Local Jupyter Environment

Install the required packages:

```bash
pip install tensorflow matplotlib
```

Then open the notebook using Jupyter Notebook or JupyterLab.

## 📁 Repository Contents

```text
01_Data_Preprocessing/
│
├── Assignment_01_Data_Preprocessing_MNIST.ipynb
├── Problem_Statement.pdf
└── README.md
```

## ✅ Learning Outcomes

After completing this practical, the following concepts are demonstrated:

- Dataset loading with TensorFlow/Keras
- Image-data preprocessing
- Pixel normalization
- Train-test separation
- Basic data visualization
- Neural-network input preparation
- Model training and evaluation
- Saving a trained Keras model

## 📄 Academic Information

- **Subject:** Deep Learning
- **Practical:** 01
- **Semester:** 5
- **Academic Year:** 2026–27
- **Department:** CSE-AI
- **Division:** E
