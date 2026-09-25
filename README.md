# LSTM Stock Price Prediction

An end-to-end deep learning project for time-series stock price prediction using Long Short-Term Memory (LSTM) neural networks. It covers data acquisition, leakage-safe preprocessing, sequence generation, LSTM training, chronological evaluation, baseline comparison, visualization, and recursive future forecasting.

> Educational and research use only. Forecasts are model estimates, not financial advice or guarantees of future market performance.

## Features

- Yahoo Finance data download with `yfinance`
- Local CSV input for offline/reproducible experiments
- Chronological train / validation / test split
- Training-only standardization to prevent data leakage
- Sliding-window LSTM sequence generation
- Configurable stacked PyTorch LSTM
- AdamW, Huber loss, gradient clipping, ReduceLROnPlateau, and early stopping
- MAE, RMSE, MAPE, sMAPE, R², and directional accuracy
- Persistence baseline for honest comparison
- Recursive multi-step future forecasting
- Training, prediction, and forecast plots
- Timestamped experiment artifacts
- Synthetic stock-like dataset generator
- Unit tests and GitHub Actions CI

## Project structure

```text
.
├── configs/
│   ├── default.yaml
│   └── fast.yaml
├── docs/
│   └── methodology.md
├── scripts/
│   └── generate_synthetic_data.py
├── src/lstm_stock/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── forecast.py
│   ├── model.py
│   ├── pipeline.py
│   ├── training.py
│   └── visualize.py
├── tests/
│   ├── test_data.py
│   ├── test_forecast.py
│   └── test_metrics.py
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Quick start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

### Train on a real stock

The default configuration uses AAPL:

```bash
lstm-stock --config configs/default.yaml
```

Override the ticker:

```bash
lstm-stock --config configs/default.yaml --ticker MSFT
```

Override the future forecast horizon:

```bash
lstm-stock --config configs/default.yaml --ticker NVDA --horizon 20
```

### Fully offline demo

```bash
python scripts/generate_synthetic_data.py --output data/synthetic.csv
lstm-stock --config configs/fast.yaml --csv data/synthetic.csv
```

### Run tests

```bash
pytest
```

Or:

```bash
make test
```

## Modeling workflow

```text
Market data / CSV
       │
       ▼
Cleaning + chronological sorting
       │
       ▼
Train / validation / test split
       │
       ▼
Fit scalers on TRAIN only
       │
       ▼
Sliding-window sequence generation
       │
       ▼
Stacked LSTM regressor
       │
       ▼
Validation-based early stopping
       │
       ├──────────────► Persistence baseline
       │
       ▼
Untouched chronological test set
       │
       ▼
Metrics + prediction plots
       │
       ▼
Recursive future forecast
```

## Leakage-safe preprocessing

The pipeline fits the feature and target `StandardScaler` objects using only the chronological training rows. Validation and test data are transformed with those already-fitted scalers.

Each sequence predicts a later target from earlier observations. Validation/test windows may contain observations from an earlier split because those observations would already be known at prediction time, but a sequence never contains data from after its target timestamp.

## LSTM architecture

The model consists of:

```text
Input sequence
    ↓
Stacked LSTM
    ↓
Last hidden representation
    ↓
LayerNorm
    ↓
Linear projection
    ↓
GELU
    ↓
Dropout
    ↓
Scalar price prediction
```

Training uses:

- Huber loss
- AdamW optimizer
- gradient clipping
- learning-rate reduction on validation plateaus
- validation early stopping
- reproducible random seeds

## Evaluation

The final test partition is never used to select epochs or tune the model.

Reported metrics:

| Metric | Purpose |
|---|---|
| MAE | Average absolute price error |
| RMSE | Penalizes larger errors more strongly |
| MAPE | Percentage error relative to actual price |
| sMAPE | Symmetric percentage error |
| R² | Variance-explanation diagnostic |
| Directional accuracy | Correct sign of next movement vs. previous observed price |

The project also evaluates a **persistence baseline**:

```text
prediction(t) = actual(t - 1)
```

This matters because price levels can be strongly autocorrelated. A neural network should be judged against a simple baseline rather than by raw RMSE alone.

## Future forecasting

Future prices are generated recursively. After the first unseen price is predicted, that prediction is inserted into the next input window to predict the following step.

For multivariate experiments, future non-target features are carried forward from their last observed values. This is deliberately transparent and should be replaced by known future covariates or separate covariate models in more advanced research.

## Configuration

Main options are in `configs/default.yaml`.

```yaml
data:
  ticker: AAPL
  target: Close
  features: [Close]
  train_ratio: 0.70
  val_ratio: 0.15

model:
  sequence_length: 60
  hidden_size: 64
  num_layers: 2
  dropout: 0.20

train:
  batch_size: 64
  epochs: 60
  learning_rate: 0.001
  patience: 10
  seed: 42
  device: auto

forecast:
  horizon: 30
```

## Local CSV format

A CSV must contain the configured date, target, and feature columns.

```csv
Date,Open,High,Low,Close,Volume
2024-01-02,100.0,102.0,99.5,101.4,1250000
2024-01-03,101.4,103.1,100.8,102.7,1180000
```

The default configuration only requires `Date` and `Close`.

## Generated artifacts

Every run creates a timestamped folder under `artifacts/` containing:

- `resolved_config.json`
- `training_history.csv`
- `metrics.json`
- `test_predictions.csv`
- `future_forecast.csv`
- `loss_curve.png`
- `test_predictions.png`
- `future_forecast.png`
- `model.pt`

## Reproducibility

Python, NumPy, and PyTorch are seeded. Exact GPU results can still vary between hardware, CUDA, and PyTorch versions.

For a research report, run multiple seeds and report the distribution of test metrics rather than relying on one training run.

## Important limitations

Stock prices are noisy, non-stationary, and affected by information not necessarily contained in historical price sequences. Good historical forecasting metrics do **not** establish a profitable trading strategy.

A trading study would additionally need transaction costs, slippage, position sizing, risk controls, regime analysis, multiple out-of-sample periods, and uncertainty estimation.

See [docs/methodology.md](docs/methodology.md) for modeling assumptions and recommended extensions.

## Research extensions

Possible next steps include:

- log-return prediction instead of price-level prediction
- multivariate OHLCV and technical indicators
- probabilistic LSTM prediction intervals
- walk-forward retraining
- conformal prediction
- ARIMA / ETS baselines
- XGBoost / LightGBM baselines
- temporal convolutional networks
- transformer forecasting models
- ensemble forecasts
- market-regime detection
- transaction-cost-aware backtesting

## License

MIT License. See [LICENSE](LICENSE).
