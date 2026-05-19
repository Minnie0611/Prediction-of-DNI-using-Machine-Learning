# Incremental Learning Framework for DNI Prediction

This project develops an incremental learning framework for predicting **Direct Normal Irradiance (DNI)** using physics-informed machine learning. The framework is built on top of an existing DNI prediction pipeline and extends it with monthly model updates, replay buffer maintenance, and sky-condition-specific neural networks.

## Project Overview

Direct Normal Irradiance (DNI) is an important solar radiation component for solar energy systems, especially concentrated solar power (CSP) and solar tracking applications. However, DNI is more difficult and expensive to measure directly compared with Global Horizontal Irradiance (GHI).  

This project predicts DNI using commonly available meteorological and solar geometry features, including:

- Global Horizontal Irradiance (GHI)
- Temperature
- Dew Point
- Atmospheric Pressure
- Clearness Index (Kt)
- Solar Cosine Zenith Angle
- Extraterrestrial Horizontal Irradiance (G0)

The main goal is to improve DNI prediction under changing weather and seasonal conditions by using an **incremental learning strategy**.

## Main Idea

Instead of training one static model once and using it forever, this framework updates the model over time as new data becomes available.

The workflow is:

1. Train a **base model** using historical data.
2. Split new incoming data into **monthly blocks**.
3. Maintain a **replay buffer** for each sky condition.
4. Fine-tune the previous model checkpoint using the current monthly data plus replay samples.
5. Evaluate monthly performance and final aggregated performance.

This design helps the model adapt to new patterns while reducing catastrophic forgetting.

## Dataset

The dataset comes from the **National Solar Radiation Database (NSRDB)** and covers solar and meteorological measurements for Bethlehem, Pennsylvania.

The notebook combines data from multiple years and uses the following temporal structure:

- Historical data for base training
- Monthly data blocks for incremental learning
- Later-year data for final testing

Default configuration:

```python
BASE_TRAIN_END = "2021-12-31 23:59:59"
INCREMENTAL_START = "2022-01-01 00:00:00"
TEST_START = "2023-01-01 00:00:00"
```

## Feature Engineering

The preprocessing pipeline creates several physics-related features:

### 1. Timestamp Construction

The original time columns are combined into a single timestamp:

```python
["Year", "Month", "Day", "Hour", "Minute"]
```

### 2. Solar Geometry Features

The framework computes:

- Day of year
- Solar zenith angle
- Solar cosine zenith angle
- Extraterrestrial irradiance
- Horizontal extraterrestrial irradiance `G0`

### 3. Clearness Index

The clearness index is calculated as:

```text
Kt = GHI / G0
```

This value is used to classify sky conditions.

### 4. Daytime Filtering

Nighttime samples are removed before training because DNI prediction is only meaningful when solar radiation is available.

## Sky Condition Classification

The framework trains separate models for different sky conditions. The categories are based on the clearness index:

| Sky Condition | Rule |
|---|---|
| Overcast | `Kt <= 0.35` |
| Partly Cloudy | `0.35 < Kt <= 0.70` |
| Clear Sky | `Kt > 0.70` |

This allows each model to specialize in a different irradiance pattern.

## Model Architecture

Each sky condition uses an independent neural network with the same structure:

```text
Input Layer
Dense(128) + Batch Normalization + ReLU
Dense(64) + Batch Normalization + ReLU
Dense(32) + Batch Normalization + ReLU
Dense(1)
```

The model is trained with:

- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- Metrics: MAE and MSE
- Early stopping
- Model checkpointing

## Physics-Based Constraint

To improve physical consistency, predictions are constrained by the relationship between GHI and DNI:

```text
0 <= DNI <= GHI / cos(theta_z)
```

This prevents the model from generating physically unrealistic DNI values.

## Incremental Learning Framework

### Base Training

The base model is trained using historical daytime data before the incremental learning period. Each sky condition has:

- One scaler
- One neural network model
- One checkpoint
- One replay buffer

### Monthly Update

For each month, the framework:

1. Loads the monthly data block.
2. Splits the block by sky condition.
3. Samples representative data from the replay buffer.
4. Combines replay data with new monthly data.
5. Loads the previous checkpoint.
6. Fine-tunes the model for fewer epochs.
7. Saves the updated checkpoint.
8. Updates the replay buffer.
9. Evaluates monthly performance.

### Replay Buffer

Each sky condition has its own replay buffer. The buffer stores representative historical samples and prevents the model from forgetting older patterns.

The replay buffer uses:

- Fixed maximum capacity
- Bucket-based sampling by DNI level
- Recent data retention
- Stratified sampling for diversity

Default parameters:

```python
REPLAY_CAPACITY_PER_CONDITION = 12000
REPLAY_SAMPLE_SIZE_PER_UPDATE = 3000
INCREMENTAL_EPOCHS = 15
BASE_EPOCHS = 100
BATCH_SIZE = 32
```

## Evaluation Metrics

The project evaluates model performance using:

- **R² Score**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**

Performance is evaluated in two ways:

1. Monthly performance by sky condition
2. Final aggregated performance across all three conditions

## Output Files

The framework saves outputs into:

```text
incremental_outputs/
```

Generated files include:

| File | Description |
|---|---|
| `base_model_<condition>.keras` | Base model checkpoint for each sky condition |
| `incremental_model_<condition>_<month>.keras` | Updated monthly checkpoint |
| `monthly_incremental_report.csv` | Monthly metrics by condition |
| `final_test_predictions.csv` | Final test predictions with actual and predicted DNI |

## Visualization

The notebook includes optional plots for:

- Monthly MAE by sky condition
- Actual vs. predicted DNI on the test set

These visualizations help evaluate whether incremental updates improve stability over time.

## Project Structure

A recommended GitHub structure is:

```text
.
├── README.md
├── dni_incremental_learning_framework.ipynb
├── incremental_outputs/
│   ├── monthly_incremental_report.csv
│   ├── final_test_predictions.csv
│   └── model checkpoints
├── requirements.txt
└── data/
    └── raw or processed data files
```

## How to Run

1. Clone the repository.

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Open the notebook.

```bash
jupyter notebook dni_incremental_learning_framework.ipynb
```

4. Run the notebook from top to bottom.

5. Check outputs in:

```text
incremental_outputs/
```

## Requirements

Main Python packages:

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
keras
jupyter
```

## Key Contributions

This project contributes an adaptive DNI prediction pipeline with the following strengths:

- Uses physics-informed feature engineering.
- Separates data into sky-condition-specific models.
- Applies physical constraints to prevent unrealistic predictions.
- Introduces monthly incremental updates.
- Uses replay buffers to reduce catastrophic forgetting.
- Evaluates both condition-level and aggregated performance.

## Future Work

Possible improvements include:

- Compare incremental learning against a static baseline model.
- Test different replay buffer strategies.
- Add concept drift detection.
- Explore alternative models such as LightGBM, XGBoost, or LSTM.
- Evaluate performance under seasonal and extreme-weather conditions.
- Deploy the final model in a Streamlit dashboard.

## Acknowledgement

This project is related to solar irradiance prediction research at Lehigh University and focuses on improving DNI prediction for solar energy applications through physics-informed machine learning and incremental learning.
