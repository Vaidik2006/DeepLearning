# Deep Learning Practicals

A structured collection of Deep Learning practical assignments covering data preprocessing, neural networks, sequence models, CNNs, and transfer learning.

## Assignment Index

| No. | Assignment | Main Topic | Dataset / Application | Framework |
|---|---|---|---|---|
| **01** | [Data Preprocessing & Visualization](Assignment1/) | Preprocessing, normalization, visualization | MNIST | TensorFlow / Keras |
| **02** | [MLP Wine Quality Classification](Assignment2/) | Multilayer Perceptron (MLP) | Wine Quality (`WineQT.csv`) | TensorFlow / Keras |
| **03** | [Forward & Backpropagation + Learning Rate Analysis](Assignment3/) | Forward propagation, backpropagation, hyperparameter analysis | MNIST | TensorFlow / Keras |
| **04** | [LSTM Weather Forecasting](Assignment4/) | Time-series forecasting with LSTM | Jena Climate Dataset | TensorFlow / Keras |
| **05** | [RNN, LSTM & GRU Sequence Classification](Assignment5/) | Recurrent neural networks and sequence classification | Sequence classification | TensorFlow / Keras |
| **06** | [CNN Tomato Disease Classification](Assignment6/) | Convolutional Neural Network image classification | Tomato Disease Dataset | TensorFlow / Keras |
| **07** | [Transfer Learning for Grapevine Classification](Assignment7/) | Transfer learning and model comparison | Grapevine Leaves Dataset | PyTorch / Torchvision |

---

## Repository Structure

```text
DeepLearning/
├── Assignment1/
├── Assignment2/
├── Assignment3/
├── Assignment4/
├── Assignment5/
├── Assignment6/
├── Assignment7/
└── README.md
```

Each assignment folder contains its own README with the problem statement, objective, dataset, methodology, model/technique, results, technologies, and run instructions.

> Large datasets, trained model weights, virtual environments, and cache files are excluded from version control using `.gitignore`.

---

## Topics Covered

### 01 — Data Preprocessing & Visualization
MNIST preprocessing, normalization, visualization, and basic neural-network classification.

### 02 — Multilayer Perceptron
Wine Quality classification using feature scaling, stratified splitting, an MLP, dropout, accuracy, and confusion-matrix evaluation.

### 03 — Forward & Backpropagation
Forward propagation, backpropagation, and analysis of different learning rates and epoch counts.

### 04 — LSTM Weather Forecasting
Time-series forecasting using sequential weather observations and an LSTM model.

### 05 — RNN, LSTM & GRU
Sequence classification using recurrent architectures and comparison of RNN, LSTM, and GRU approaches.

### 06 — CNN Tomato Disease Classification
A three-block CNN for 10-class tomato leaf disease classification using convolution, pooling, dropout, and softmax classification.

### 07 — Transfer Learning
Comparison of pretrained AlexNet, VGG16, ResNet50, and EfficientNetB0 models for Grapevine Leaves classification.

---

## Technologies

- Python
- TensorFlow / Keras
- PyTorch / Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Google Colab

---

## Running the Assignments

Notebook-based assignments can be opened in Jupyter Notebook or Google Colab.

1. Open the required assignment folder.
2. Install dependencies if a `requirements.txt` file is provided.
3. Place the required dataset at the path specified by the notebook.
4. Run the notebook cells sequentially.

For Assignment 7:

```bash
pip install -r Assignment7/requirements.txt
```

Then run the required training script.

---

## Dataset Policy

Large datasets and trained model weights are intentionally excluded from GitHub. The repository focuses on the source code, notebooks, documentation, reproducible workflows, and results.

---

Deep Learning Practicals  
Academic Year 2026–27
