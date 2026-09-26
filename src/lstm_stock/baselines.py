from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class BenchmarkResult:
    model: str
    mae: float
    rmse: float


def persistence(y_previous: np.ndarray) -> np.ndarray:
    return np.asarray(y_previous, dtype=float).reshape(-1)


def moving_average_forecast(history: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(history, dtype=float).reshape(-1)
    if window < 1:
        raise ValueError("window must be positive.")
    if len(values) < window:
        raise ValueError("history is shorter than window.")
    outputs = []
    for index in range(window, len(values)):
        outputs.append(values[index - window:index].mean())
    return np.asarray(outputs, dtype=float)


def flatten_sequences(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError("x must have shape [samples, sequence, features].")
    return x.reshape(x.shape[0], -1)


def fit_ridge_baseline(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    model = Ridge(alpha=alpha)
    model.fit(flatten_sequences(train_x), np.asarray(train_y).reshape(-1))
    return model.predict(flatten_sequences(test_x))


def fit_random_forest_baseline(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    n_estimators: int = 300,
    random_state: int = 42,
) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(flatten_sequences(train_x), np.asarray(train_y).reshape(-1))
    return model.predict(flatten_sequences(test_x))


def score_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> BenchmarkResult:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return BenchmarkResult(
        model=name,
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )
