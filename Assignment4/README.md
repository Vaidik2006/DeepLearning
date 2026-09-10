# Assignment 04 — LSTM Weather Time-Series Forecasting

## Problem Statement

Develop an LSTM-based model for time-series forecasting using a stock price, weather, or sales dataset.

## Objective

Develop an LSTM model to forecast the next-hour air temperature from historical weather observations. The practical demonstrates sequence preparation, chronological splitting, normalization, LSTM modeling, training, and regression-based evaluation.

## Dataset

**Dataset:** Jena Climate weather dataset

The supplied CSV contains **420,551 observations** and **15 columns**, recorded at **10-minute intervals** from **01 January 2009 00:10** to **01 January 2017 00:00**. The dataset contains **0 missing values**.

For the practical, the following variables are used:

- `T (degC)` — air temperature, target variable
- `p (mbar)` — atmospheric pressure
- `rh (%)` — relative humidity
- `wv (m/s)` — wind velocity

The original 10-minute observations are aggregated to **hourly averages** before sequence generation.

## Methodology / Workflow

1. Load the Jena Climate CSV dataset.
2. Parse and sort the date-time column.
3. Select relevant weather features.
4. Resample the 10-minute observations to hourly frequency.
5. Split the data chronologically into 80% training and 20% testing sets.
6. Fit `StandardScaler` only on the training data.
7. Create 24-hour sliding windows.
8. Train an LSTM regression model.
9. Evaluate predictions using MAE and RMSE.
10. Visualize training loss and actual-vs-predicted temperatures.

## Model / Technique

**Architecture:**

`Input (24 × 4) → LSTM (64) → Dense (32, ReLU) → Dense (1)`

**Training configuration:**

- Optimizer: Adam
- Learning rate: 0.001
- Loss: Mean Squared Error (MSE)
- Epochs: 10
- Batch size: 64
- Validation split: 10%
- Shuffle: disabled to preserve temporal order

The model receives the previous **24 hourly observations** and predicts the temperature for the next hour.

## Results

The notebook calculates the following test-set metrics after training:

- **MAE (Mean Absolute Error):** generated when the notebook is executed
- **RMSE (Root Mean Squared Error):** generated when the notebook is executed

The notebook also produces:

- Training vs. validation loss curve
- Actual vs. predicted temperature plot
- Sample prediction table

**Note:** Final model metrics are intentionally not hard-coded because TensorFlow execution is environment-dependent. Running the notebook produces the authoritative results for the selected environment.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook / Google Colab

## How to Run

### Google Colab

1. Upload `Assignment_04_LSTM_Weather_Forecasting.ipynb` and `jena_climate_2009_2016.csv` to Google Colab.
2. Open the notebook.
3. Run all cells from top to bottom.
4. Review the generated MAE, RMSE, training curve, and prediction plot.

Then launch Jupyter:

```bash
jupyter notebook
```

Open `Assignment_04_LSTM_Weather_Forecasting.ipynb` and run all cells.

### Repository Contents

```text
04_LSTM_Weather_Forecasting/
├── Assignment_04_LSTM_Weather_Forecasting.ipynb
├── jena_climate_2009_2016.csv
└── README.md
```
