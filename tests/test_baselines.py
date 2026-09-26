import numpy as np

from lstm_stock.baselines import (
    fit_ridge_baseline,
    flatten_sequences,
    moving_average_forecast,
    score_model,
)


def test_flatten_sequences():
    x = np.arange(24, dtype=float).reshape(4, 3, 2)
    flat = flatten_sequences(x)
    assert flat.shape == (4, 6)


def test_moving_average_forecast():
    values = np.array([1, 2, 3, 4, 5], dtype=float)
    result = moving_average_forecast(values, window=3)
    assert np.allclose(result, [2.0, 3.0])


def test_ridge_baseline_and_score():
    x_train = np.arange(60, dtype=float).reshape(10, 3, 2)
    y_train = np.linspace(0, 1, 10)
    x_test = x_train[:3]
    pred = fit_ridge_baseline(x_train, y_train, x_test)
    result = score_model("ridge", y_train[:3], pred)
    assert pred.shape == (3,)
    assert result.mae >= 0.0
    assert result.rmse >= 0.0
