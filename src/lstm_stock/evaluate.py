from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def predict_scaled(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    model.eval()
    chunks = []
    with torch.no_grad():
        for (batch_x,) in loader:
            chunks.append(model(batch_x.to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(chunks)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, previous_actual: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true - previous_actual) == np.sign(y_pred - previous_actual)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, previous_actual: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    denominator = np.clip(np.abs(y_true), 1e-8, None)
    smape_denominator = np.clip(np.abs(y_true) + np.abs(y_pred), 1e-8, None)
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_percent": float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0),
        "smape_percent": float(np.mean(2.0 * np.abs(y_pred - y_true) / smape_denominator) * 100.0),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if previous_actual is not None:
        previous_actual = np.asarray(previous_actual, dtype=float).reshape(-1)
        metrics["directional_accuracy"] = directional_accuracy(y_true, y_pred, previous_actual)
    return metrics
