from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_previous_raw: np.ndarray
    test_timestamps: pd.DatetimeIndex
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    full_scaled_features: np.ndarray
    cleaned_frame: pd.DataFrame
    target_feature_index: int


def download_market_data(ticker: str, start: str, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if frame.empty:
        raise ValueError(f"No market data returned for ticker '{ticker}'.")

    if isinstance(frame.columns, pd.MultiIndex):
        last_level = frame.columns.get_level_values(-1)
        if ticker in last_level:
            frame = frame.xs(ticker, axis=1, level=-1)
        else:
            frame.columns = [str(column[0]) for column in frame.columns]

    return frame.reset_index()


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def clean_frame(frame: pd.DataFrame, date_column: str, features: list[str], target: str) -> pd.DataFrame:
    required = list(dict.fromkeys([date_column, *features, target]))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = frame[required].copy()
    clean[date_column] = pd.to_datetime(clean[date_column], errors="coerce", utc=False)
    numeric_columns = list(dict.fromkeys([*features, target]))
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(subset=[date_column]).sort_values(date_column)
    clean = clean.drop_duplicates(subset=[date_column], keep="last")
    clean[numeric_columns] = clean[numeric_columns].ffill()
    clean = clean.dropna(subset=numeric_columns).set_index(date_column)

    if len(clean) < 10:
        raise ValueError("Dataset is too small after cleaning.")
    return clean


def _make_sequences(scaled_features: np.ndarray, scaled_target: np.ndarray, sequence_length: int):
    x_values, y_values, target_rows = [], [], []
    for target_row in range(sequence_length, len(scaled_features)):
        x_values.append(scaled_features[target_row - sequence_length:target_row])
        y_values.append(float(scaled_target[target_row, 0]))
        target_rows.append(target_row)
    return (
        np.asarray(x_values, dtype=np.float32),
        np.asarray(y_values, dtype=np.float32).reshape(-1, 1),
        np.asarray(target_rows, dtype=np.int64),
    )


def prepare_datasets(
    frame: pd.DataFrame,
    date_column: str,
    features: list[str],
    target: str,
    sequence_length: int,
    train_ratio: float,
    val_ratio: float,
) -> DatasetBundle:
    clean = clean_frame(frame, date_column, features, target)
    n_rows = len(clean)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    if train_end <= sequence_length:
        raise ValueError("Training partition is too short for sequence_length.")
    if val_end <= train_end or val_end >= n_rows:
        raise ValueError("Chronological split produced an empty validation or test partition.")

    feature_values = clean[features].to_numpy(dtype=np.float64)
    target_values = clean[[target]].to_numpy(dtype=np.float64)
    feature_scaler = StandardScaler().fit(feature_values[:train_end])
    target_scaler = StandardScaler().fit(target_values[:train_end])

    scaled_features = feature_scaler.transform(feature_values)
    scaled_target = target_scaler.transform(target_values)
    all_x, all_y, target_rows = _make_sequences(scaled_features, scaled_target, sequence_length)

    train_mask = target_rows < train_end
    val_mask = (target_rows >= train_end) & (target_rows < val_end)
    test_mask = target_rows >= val_end
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("At least one generated sequence is required in every split.")

    test_rows = target_rows[test_mask]
    previous_raw = target_values[test_rows - 1, 0]

    return DatasetBundle(
        train_x=all_x[train_mask], train_y=all_y[train_mask],
        val_x=all_x[val_mask], val_y=all_y[val_mask],
        test_x=all_x[test_mask], test_y=all_y[test_mask],
        test_previous_raw=previous_raw.astype(np.float64),
        test_timestamps=pd.DatetimeIndex(clean.index[test_rows]),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        full_scaled_features=scaled_features.astype(np.float32),
        cleaned_frame=clean,
        target_feature_index=features.index(target),
    )


def inverse_target(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return scaler.inverse_transform(array).reshape(-1)
