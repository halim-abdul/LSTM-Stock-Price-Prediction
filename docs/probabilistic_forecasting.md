# Probabilistic Forecasting with Monte Carlo Dropout

Point forecasts hide uncertainty. This branch adds Monte Carlo (MC) dropout utilities that repeatedly evaluate the trained LSTM while dropout remains active.

The resulting distribution of predictions can be summarized with empirical quantiles, for example the 5th and 95th percentiles.

## Core workflow

```text
same input window
      │
      ├─ dropout mask 1 → prediction 1
      ├─ dropout mask 2 → prediction 2
      ├─ dropout mask 3 → prediction 3
      └─ ...
             │
             ▼
 empirical prediction distribution
             │
             ├─ mean
             ├─ 5% quantile
             └─ 95% quantile
```

## API

```python
from lstm_stock.uncertainty import mc_dropout_predict, prediction_interval

draws = mc_dropout_predict(
    model,
    x_test,
    device=device,
    samples=200,
)

mean, low, high = prediction_interval(
    draws,
    lower=0.05,
    upper=0.95,
)
```

## Coverage

`interval_coverage` reports the fraction of realized targets that fall inside a prediction interval.

Coverage alone is not enough. Very wide intervals can achieve high coverage while being uninformative. A stronger study should report both coverage and interval width.

## Important interpretation

MC dropout provides an approximate uncertainty signal rather than a guaranteed calibrated probability interval. Financial time series can undergo distribution shifts, so historical interval behavior may not transfer to future regimes.

Recommended extensions:

- conformal prediction
- quantile loss
- deep ensembles
- Bayesian recurrent networks
- heteroscedastic likelihood models
- interval calibration by market regime
