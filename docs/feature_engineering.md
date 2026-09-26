# Feature Engineering Branch

This branch adds a causal technical-feature pipeline for OHLCV market data.

## Included signals

- one-period percentage and log returns
- SMA(5), SMA(20)
- EMA(12), EMA(26)
- MACD and MACD signal
- 20-period realized return volatility
- Bollinger z-score
- RSI(14)
- ATR(14)
- rolling volume z-score
- price-to-SMA distance
- intraday range fraction
- cyclical day-of-week encoding

Every rolling or exponentially weighted feature uses the current or earlier rows only. No centered rolling windows or backward filling are used.

## Build a dataset from Yahoo Finance

```bash
python scripts/build_feature_dataset.py \
  --ticker AAPL \
  --start 2015-01-01 \
  --output data/aapl_features.csv
```

Then train:

```bash
lstm-stock \
  --config configs/technical_features.yaml \
  --csv data/aapl_features.csv
```

## Build from an existing OHLCV CSV

```bash
python scripts/build_feature_dataset.py \
  --csv data/raw_prices.csv \
  --output data/technical_features.csv
```

## Why this is separate from main

Technical indicators can help representation learning, but they can also add redundant and highly correlated inputs. Keeping this work on a separate branch makes it easy to compare a clean univariate LSTM against a multivariate technical-feature experiment.

For a rigorous report, compare both branches on the same date range, split boundaries, random seeds, and evaluation metrics.
