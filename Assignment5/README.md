# Assignment 05 — RNN, LSTM and GRU Sequence Classification

## Problem Statement

Implement and compare RNN, LSTM, and GRU models for sequence classification, and analyze their performance using appropriate evaluation metrics.

## Objective

Implement three recurrent neural network architectures for binary sentiment classification and compare them using accuracy, precision, recall, F1-score, test loss, and training time.

## Dataset

**IMDB Movie Reviews** — 50,000 labeled reviews: 25,000 training and 25,000 testing samples. Labels are Negative (0) and Positive (1). The experiment uses the 10,000 most frequent words and pads/truncates each review to 200 tokens.

The dataset is loaded directly through `tf.keras.datasets.imdb`; no separate dataset file is required.

## Methodology / Workflow

1. Load the IMDB dataset.
2. Limit vocabulary to 10,000 words.
3. Pad/truncate sequences to 200 tokens.
4. Build Simple RNN, LSTM and GRU models.
5. Keep embedding size, units and training configuration identical.
6. Train each model.
7. Evaluate on the held-out test set.
8. Compare Accuracy, Precision, Recall, F1-score, Test Loss and Training Time.
9. Plot validation accuracy.
10. Generate a confusion matrix and classification report for the best model.

## Model / Technique

```text
Simple RNN: Input → Embedding(64) → SimpleRNN(64) → Dense(1, Sigmoid)
LSTM:       Input → Embedding(64) → LSTM(64) → Dense(1, Sigmoid)
GRU:        Input → Embedding(64) → GRU(64) → Dense(1, Sigmoid)
```

Common configuration:

- Adam optimizer
- Learning rate: 0.001
- Binary cross-entropy loss
- 5 epochs
- Batch size: 128
- Validation split: 10%

## Results

The notebook generates:

- Accuracy
- Precision
- Recall
- F1-score
- Test loss
- Training time
- Validation-accuracy comparison
- Confusion matrix
- Classification report

Final numerical results are generated at runtime and are not fabricated or hard-coded.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook / Google Colab

## How to Run

### Google Colab

1. Upload `Assignment_05_RNN_LSTM_GRU_Sequence_Classification.ipynb`.
2. Open it in Google Colab.
3. Run all cells.
4. Keras downloads the IMDB dataset automatically.
5. Review the generated comparison table and plots.

### Repository Contents

```text
Assignment5/
├── Assignment_05_RNN_LSTM_GRU_Sequence_Classification.ipynb
├── README.md
└── requirements.txt
```