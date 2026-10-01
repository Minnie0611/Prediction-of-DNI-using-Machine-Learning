# Prediction of DNI Using Incremental Learning

This repository predicts **Direct Normal Irradiance (DNI)** from meteorological and solar-geometry features. It extends the original machine-learning notebook with sky-condition-specific neural networks, monthly incremental updates, and replay buffers.

## Experiment design

The split is strictly chronological:

| Period | Role | Samples |
|---|---|---:|
| 2011–2021 | Train the base models | 96,337 daytime records |
| 2022 | Monthly incremental learning | 8,759 daytime records |
| 2023 | Independent final test | 8,759 daytime records |

For each month in 2022, the model follows a **predict-before-update** protocol:

1. Load the checkpoint available at the end of the previous month.
2. Predict the current month and record its metrics.
3. Fine-tune with the current month's labelled data and replay samples.
4. Save the best checkpoint and add the current samples to the replay buffer.
5. Use the updated checkpoint from the next month onward.

The 2023 test set is never used for training, scaler fitting, replay, model selection, or hyperparameter tuning.

## Inputs and model

The seven model inputs are:

- Global Horizontal Irradiance (GHI)
- Temperature
- Dew point
- Atmospheric pressure
- Clearness index (`Kt`)
- Cosine of the solar zenith angle
- Extraterrestrial horizontal irradiance (`G0`)

Samples are divided into three sky conditions:

| Sky condition | Rule |
|---|---|
| Overcast | `Kt <= 0.35` |
| Partly cloudy | `0.35 < Kt <= 0.70` |
| Clear sky | `Kt > 0.70` |

Each condition uses an independent dense neural network with hidden layers of 128, 64, and 32 units, Batch Normalization, and ReLU activation. The base scaler remains fixed during incremental updates. Each sky condition also has a replay buffer with a capacity of 12,000 samples; at most 3,000 replay samples are drawn per monthly update.

Predictions are clipped to the physical range:

```text
0 <= DNI <= GHI / cos(theta_z)
```

## 2023 independent-test results

| Model | R² | MAE (W/m²) | RMSE (W/m²) |
|---|---:|---:|---:|
| Base model | 0.9615 | 41.44 | 66.32 |
| Incremental model | 0.9654 | 39.52 | 62.83 |

The incremental model reduced overall MAE by **4.64%** and RMSE by **5.26%**.

Performance varied by weather condition:

- **Clear sky:** the physical relationship between DNI, GHI, `Kt`, and solar geometry is stable. The base model was already strong (`R² = 0.9404`), so the incremental improvement was small but consistent; MAE fell by 3.26%.
- **Partly cloudy:** recent data corrected a systematic underestimation. Mean bias improved from −26.19 to −12.48 W/m², 57.2% of samples had lower absolute error, and MAE fell by 7.41%. This remained the most difficult category in absolute terms, with an incremental-model MAE of 64.61 W/m².
- **Overcast:** MAE increased by 10.55%, while RMSE decreased by 3.43%. The model reduced several large high-DNI errors but increased smaller errors among the many zero and low-DNI samples.

Detailed tables and figures are available in [`results/`](results/). A Chinese explanation is available in [`results/实验结果说明.md`](results/%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C%E8%AF%B4%E6%98%8E.md).

## Repository structure

```text
.
├── dni_incremental_learning_framework.ipynb
├── scripts/
│   ├── check_experiment.py
│   ├── run_notebook.py
│   ├── update_notebook.py
│   └── verify_results.py
├── results/
│   ├── final_test_comparison.csv
│   ├── monthly_incremental_report.csv
│   ├── run_summary.json
│   └── figures and Chinese result notes
├── requirements.txt
└── .gitignore
```

## Run the notebook

```bash
git clone https://github.com/Minnie0611/Prediction-of-DNI-using-Machine-Learning.git
cd Prediction-of-DNI-using-Machine-Learning
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Run interactively:

```bash
jupyter notebook dni_incremental_learning_framework.ipynb
```

Or execute the notebook from the command line while preserving outputs:

```bash
python scripts/run_notebook.py
```

The notebook downloads the published yearly CSV files on first use and caches them in `data_cache/`. Generated checkpoints, replay buffers, predictions, and reports are written to `incremental_outputs/`.

## Validation scripts

```bash
python scripts/check_experiment.py
python scripts/verify_results.py
```

`check_experiment.py` checks the chronological boundaries, replay-buffer behavior, and predict-before-update order. `verify_results.py` recomputes saved metrics and checks model state after a complete run; it therefore requires the generated files in `incremental_outputs/`.

## Current limitations

- Results are from one site and one random seed.
- Solar zenith angle uses the simplified formula retained from the original notebook.
- The reported improvement combines the effects of new data, replay, and fine-tuning settings; an ablation study is still needed to isolate each contribution.
- Overcast performance needs a dedicated strategy, such as zero-DNI classification followed by positive-DNI regression, robust loss functions, and replay-distribution experiments.
