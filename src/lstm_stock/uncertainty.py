from __future__ import annotations

import numpy as np
import torch
from torch import nn


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers while leaving the rest of the model in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    samples: int = 100,
) -> np.ndarray:
    if samples < 2:
        raise ValueError("samples must be at least 2.")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, sequence, features].")

    enable_dropout(model)
    tensor = torch.from_numpy(x).to(device)
    draws = []
    with torch.no_grad():
        for _ in range(samples):
            draws.append(model(tensor).detach().cpu().numpy().reshape(-1))

    model.eval()
    return np.stack(draws, axis=0)


def prediction_interval(
    draws: np.ndarray,
    lower: float = 0.05,
    upper: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError("draws must have shape [samples, observations].")
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("Require 0 < lower < upper < 1.")

    mean = draws.mean(axis=0)
    low = np.quantile(draws, lower, axis=0)
    high = np.quantile(draws, upper, axis=0)
    return mean, low, high


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if not (y_true.shape == lower.shape == upper.shape):
        raise ValueError("Shapes must match.")
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
