import numpy as np
import torch

from lstm_stock.model import LSTMRegressor
from lstm_stock.uncertainty import interval_coverage, mc_dropout_predict, prediction_interval


def test_mc_dropout_shape_and_interval_order():
    x = np.ones((4, 8, 1), dtype=np.float32)
    model = LSTMRegressor(input_size=1, hidden_size=8, num_layers=2, dropout=0.5)
    draws = mc_dropout_predict(model, x, torch.device("cpu"), samples=20)

    assert draws.shape == (20, 4)
    mean, low, high = prediction_interval(draws)
    assert mean.shape == (4,)
    assert np.all(low <= high)


def test_interval_coverage():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    low = np.array([0.0, 1.5, 3.1, 3.0])
    high = np.array([1.5, 2.5, 3.9, 4.5])
    assert interval_coverage(y, low, high) == 0.75
