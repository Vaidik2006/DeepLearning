# Assignment 02 — Multilayer Perceptron for Wine Quality Classification

## Problem Statement

Design and implement a **Multilayer Perceptron (MLP)** for classification of the **Wine** dataset and evaluate its performance using **accuracy** and a **confusion matrix**.

## Objective

- Load and inspect the Wine Quality dataset.
- Prepare the input features and target variable.
- Split the dataset into training and testing subsets.
- Standardize numerical features.
- Design and train an MLP using TensorFlow/Keras.
- Evaluate the classifier using test accuracy.
- Analyze predictions using a confusion matrix.

## Dataset

### Wine Quality Dataset

The supplied `WineQT.csv` contains **1,143 wine samples and 13 columns**. The `quality` column is treated as the target variable and contains six observed classes: **3, 4, 5, 6, 7, and 8**.

The `Id` column is an identifier and is excluded from the model input.

| Property | Value |
|---|---:|
| Samples | 1,143 |
| Columns | 13 |
| Input features used | 11 |
| Target | `quality` |
| Classes | 3, 4, 5, 6, 7, 8 |
| Missing values | 0 |
| Test size | 20% |

## Methodology / Workflow

```text
Wine Quality Dataset
        ↓
Load & Inspect Data
        ↓
Remove Identifier (Id)
        ↓
Separate Features & Target
        ↓
Stratified 80:20 Train-Test Split
        ↓
Standardize Features
        ↓
Encode Target Classes
        ↓
Build MLP
        ↓
Train Model
        ↓
Evaluate Accuracy
        ↓
Generate Confusion Matrix
```

## MLP Architecture

```text
Input Layer
    ↓
Dense — 64 neurons, ReLU
    ↓
Dropout — 20%
    ↓
Dense — 32 neurons, ReLU
    ↓
Dense — 6 neurons, Softmax
    ↓
Wine Quality Class
```

### Training Configuration

| Parameter | Setting |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 50 |
| Batch size | 32 |
| Validation split | 20% |
| Random state | 42 |

## Evaluation

The notebook reports:

- **Test accuracy** — overall classification performance.
- **Confusion matrix** — class-by-class prediction distribution.
- **Classification report** — precision, recall, and F1-score.
- **Training/validation curves** — model behavior across epochs.

> Exact accuracy and confusion-matrix values are generated when the notebook is executed. They are intentionally not hard-coded into this README.

## Technologies

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Google Colab / Jupyter Notebook

## How to Run

### Google Colab

1. Upload `Assignment_02_MLP_Wine_Quality_Classification.ipynb` and `WineQT.csv` to Google Colab.
2. Make sure `WineQT.csv` is in the notebook's working directory.
3. Select **Runtime → Run all**.
4. Review the dataset inspection, training output, accuracy, confusion matrix, and training curves.

### Local Environment

Install the dependencies:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

Then open the notebook with Jupyter Notebook/JupyterLab.

## Repository Contents

```text
Assignment2/
│
├── Assignment_02_MLP_Wine_Quality_Classification.ipynb
├── WineQT.csv
└── README.md
```
