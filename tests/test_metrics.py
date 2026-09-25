import numpy as np

from lstm_stock.evaluate import directional_accuracy, regression_metrics


def test_regression_metrics_for_perfect_prediction():
    y = np.array([10.0, 11.0, 12.0])
    metrics = regression_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_directional_accuracy_uses_previous_observation():
    previous = np.array([10.0, 10.0, 10.0, 10.0])
    actual = np.array([11.0, 9.0, 12.0, 8.0])
    predicted = np.array([12.0, 8.0, 9.0, 7.0])
    assert directional_accuracy(actual, predicted, previous) == 0.75
