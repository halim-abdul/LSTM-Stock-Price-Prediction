from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


def recursive_forecast(
    model: nn.Module,
    last_window_scaled: np.ndarray,
    horizon: int,
    target_feature_index: int,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    device: torch.device,
) -> np.ndarray:
    if horizon < 0:
        raise ValueError("horizon cannot be negative.")
    if horizon == 0:
        return np.empty(0, dtype=float)

    window = np.asarray(last_window_scaled, dtype=np.float32).copy()
    if window.ndim != 2:
        raise ValueError("last_window_scaled must have shape [sequence_length, features].")

    predictions = []
    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            pred_scaled = float(model(tensor).cpu().item())
            pred_raw = float(target_scaler.inverse_transform([[pred_scaled]])[0, 0])
            predictions.append(pred_raw)

            next_row = window[-1].astype(np.float64).copy()
            next_row[target_feature_index] = (
                pred_raw - float(feature_scaler.mean_[target_feature_index])
            ) / float(feature_scaler.scale_[target_feature_index])
            window = np.vstack([window[1:], next_row.astype(np.float32)])

    return np.asarray(predictions, dtype=float)


def future_timestamps(last_timestamp: pd.Timestamp, horizon: int, interval: str) -> pd.DatetimeIndex:
    last_timestamp = pd.Timestamp(last_timestamp)
    if interval == "1d":
        return pd.bdate_range(last_timestamp + pd.offsets.BDay(1), periods=horizon)
    if interval == "1wk":
        return pd.date_range(last_timestamp + pd.offsets.Week(1), periods=horizon, freq="W-FRI")
    if interval == "1mo":
        return pd.date_range(last_timestamp + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")
    return pd.date_range(last_timestamp, periods=horizon + 1, freq="D")[1:]
