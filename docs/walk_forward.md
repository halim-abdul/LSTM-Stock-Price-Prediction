# Walk-Forward Backtesting

This branch adds expanding-window backtesting utilities.

A single chronological train/test split can hide regime sensitivity. Walk-forward validation repeatedly moves the forecast origin through history:

```text
Fold 1: [----------- train -----------][ test ]
Fold 2: [---------------- train ----------------][ test ]
Fold 3: [---------------------- train ----------------------][ test ]
```

## Current utility

`scripts/run_walk_forward.py` establishes the fold structure and evaluates a persistence baseline over every fold. This gives the project a repeatable benchmark before adding more expensive model retraining per fold.

Example:

```bash
python scripts/run_walk_forward.py \
  --ticker AAPL \
  --start 2015-01-01 \
  --initial-train-size 750 \
  --test-size 60 \
  --step-size 60
```

Artifacts include:

- `folds.json`
- `fold_metrics.csv`
- `predictions.csv`

## Why expanding windows?

Expanding windows resemble a realistic research setting in which all available historical data up to the current forecast origin can be used for training.

For a stronger experiment, train a fresh LSTM inside each fold using only that fold's training period, fit scalers inside the fold, and aggregate out-of-sample errors across folds.
