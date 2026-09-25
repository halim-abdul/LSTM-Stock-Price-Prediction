# Methodology

## Objective

This project performs supervised one-step-ahead regression. Given a chronological window of historical feature vectors, the LSTM predicts the target price at the next timestamp.

## Leakage control

Time-series leakage is controlled by sorting timestamps before splitting, using chronological train/validation/test partitions, fitting StandardScaler only on training rows, and assigning each sequence according to its target timestamp.

Validation and test windows may include observations from the immediately preceding partition because those observations are known at prediction time. They never include observations after their own target timestamp.

## Model

The PyTorch model uses stacked LSTM layers, layer normalization, a dense projection, GELU activation, dropout, and a scalar regression head. Training uses Huber loss, AdamW, gradient clipping, ReduceLROnPlateau, and early stopping.

## Evaluation

The untouched chronological test set reports MAE, RMSE, MAPE, sMAPE, R-squared, and directional accuracy. A persistence baseline predicts the next price as the previous observed price.

The baseline matters because price levels are often strongly autocorrelated. Low raw error alone should not be interpreted as evidence that a neural model is economically useful.

## Forecasting

Test evaluation is one-step rolling prediction from observed history. Future forecasting is recursive: each predicted target becomes part of the next model input.

For multivariate inputs, non-target future features are carried forward from their latest observed values. This is a simplifying assumption, not a statement that real future covariates are constant.

## Limitations

Historical predictive accuracy does not establish profitability. A trading study would additionally require transaction costs, slippage, position sizing, risk controls, multiple market regimes, uncertainty estimation, and out-of-sample strategy evaluation.

## Research extensions

Useful extensions include return prediction, probabilistic forecasts, rolling-window retraining, conformal intervals, technical and macroeconomic features, ARIMA and gradient-boosting baselines, temporal convolutional networks, transformers, ensembles, and regime detection.
