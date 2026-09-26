# Model Benchmarking

This branch adds non-neural baselines for comparison with the LSTM.

## Included baselines

- persistence: next price equals previous observed price
- moving average
- ridge regression on flattened sequence windows
- random forest regression on flattened sequence windows

Why this matters: a sophisticated neural network is not automatically useful simply because it fits historical data. It should be compared with simple alternatives under the exact same chronological split.

## Run

First prepare a CSV compatible with the project configuration, then:

```bash
python scripts/benchmark_models.py \
  --config configs/default.yaml \
  --csv data/synthetic.csv \
  --output artifacts/benchmarks.csv
```

The output table contains MAE and RMSE for each baseline.

## Research protocol

For a fair LSTM comparison:

1. use exactly the same raw data
2. use the same sequence length
3. use the same chronological train/test dates
4. fit preprocessing only on training data
5. compare on the identical untouched test targets
6. repeat neural training across multiple random seeds
7. report uncertainty across runs, not only the best result

Potential future baselines:

- ARIMA
- exponential smoothing
- XGBoost
- LightGBM
- temporal convolutional networks
- transformer encoders
- N-BEATS
