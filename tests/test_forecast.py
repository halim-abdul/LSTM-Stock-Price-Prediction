import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from lstm_stock.forecast import recursive_forecast


class LastValueModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, 0:1]


def test_recursive_forecast_can_repeat_last_univariate_value():
    raw = np.arange(1.0, 11.0).reshape(-1, 1)
    feature_scaler = StandardScaler().fit(raw)
    target_scaler = StandardScaler().fit(raw)
    scaled = feature_scaler.transform(raw).astype(np.float32)

    forecast = recursive_forecast(
        LastValueModel(), scaled[-4:], 3, 0, feature_scaler, target_scaler, torch.device("cpu")
    )
    assert np.allclose(forecast, [10.0, 10.0, 10.0])
