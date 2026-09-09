# Assignment 03 — Forward Propagation, Backpropagation & Learning Rate Analysis

## Problem Statement

Implement forward propagation and backpropagation using TensorFlow/Keras. Analyze the effect of different learning rates and the number of epochs on model performance.

## Objective

- Implement a neural network using TensorFlow/Keras.
- Understand forward propagation and backpropagation through the model-training process.
- Experiment with different learning rates.
- Experiment with different numbers of epochs.
- Compare model performance using accuracy and loss.
- Visualize the effect of the selected hyperparameters.

## Dataset

### MNIST Handwritten Digit Dataset

The experiment uses MNIST, a grayscale image dataset of handwritten digits from 0 to 9.

| Property | Value |
|---|---:|
| Training images | 60,000 |
| Testing images | 10,000 |
| Image dimensions | 28 × 28 |
| Number of classes | 10 |
| Pixel range | 0–255 |
| Normalized range | 0–1 |

## Methodology / Workflow

```text
Load MNIST
    ↓
Visualize Samples
    ↓
Normalize Pixel Values
    ↓
Define MLP
    ↓
Forward Propagation
    ↓
Calculate Loss
    ↓
Backpropagation + Adam Update
    ↓
Repeat for Different Learning Rates
    ↓
Repeat for Different Epoch Counts
    ↓
Evaluate Test Accuracy & Loss
    ↓
Compare Results
```

## Model Architecture

```text
Input — 28 × 28
      ↓
Flatten
      ↓
Dense — 128 neurons, ReLU
      ↓
Dense — 64 neurons, ReLU
      ↓
Dense — 10 neurons, Softmax
      ↓
Digit Prediction
```

### Training Configuration

| Parameter | Values |
|---|---|
| Optimizer | Adam |
| Learning rates | 0.1, 0.01, 0.001 |
| Epochs | 5, 10, 20 |
| Loss | Sparse Categorical Crossentropy |
| Metric | Accuracy |
| Experiments | 9 configurations |

## Results

The original submitted experiment produced the following results:

| Learning Rate | Epochs | Accuracy | Loss |
|---:|---:|---:|---:|
| 0.1 | 5 | 19.76% | 2.0116 |
| 0.1 | 10 | 18.34% | 1.9074 |
| 0.1 | 20 | 17.65% | 2.0250 |
| 0.01 | 5 | 96.52% | 0.1498 |
| 0.01 | 10 | 96.48% | 0.1785 |
| 0.01 | 20 | 96.57% | 0.2198 |
| 0.001 | 5 | 97.18% | 0.0951 |
| **0.001** | **10** | **97.93%** | **0.0819** |
| 0.001 | 20 | 97.65% | 0.1275 |

### Key Observation

The best recorded configuration was:

**Learning Rate = 0.001 | Epochs = 10 | Test Accuracy = 97.93%**

A very high learning rate of 0.1 performed poorly, while 0.001 provided stable and substantially better convergence in this experiment.

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Google Colab / Jupyter Notebook

## How to Run

### Google Colab

1. Open `Assignment_03_Forward_Backpropagation_Learning_Rate_Analysis.ipynb` in Google Colab.
2. Select **Runtime → Run all**.
3. MNIST will be downloaded automatically through Keras.
4. The nine learning-rate/epoch configurations will be trained.
5. Review the result table, comparison chart, and training curves.

### Local Environment

Install the dependencies:

```bash
pip install tensorflow numpy pandas matplotlib
```

Then open the notebook using Jupyter Notebook or JupyterLab.

## Repository Contents

```text
03_Forward_Backpropagation/
│
├── Assignment_03_Forward_Backpropagation_Learning_Rate_Analysis.ipynb
├── Problem_Statement.pdf
└── README.md
```

## Note on Evaluation

The original practical evaluates each configuration directly on the MNIST test set and also passes the test set as `validation_data` during training. The polished notebook preserves this experimental design so the documented results remain comparable to the submitted practical.
