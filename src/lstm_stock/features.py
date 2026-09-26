from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0.0, np.nan)
    return numerator / denominator


def build_technical_features(
    frame: pd.DataFrame,
    date_column: str = "Date",
    price_column: str = "Close",
) -> pd.DataFrame:
    """Create causal technical features using only present and past observations."""
    required = {date_column, "Open", "High", "Low", price_column, "Volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column]).sort_values(date_column)
    data = data.drop_duplicates(subset=[date_column], keep="last")

    numeric_columns = ["Open", "High", "Low", price_column, "Volume"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[numeric_columns] = data[numeric_columns].ffill()
    data = data.dropna(subset=numeric_columns)

    close = data[price_column].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float)

    data["return_1"] = close.pct_change()
    data["log_return_1"] = np.log(close).diff()
    data["sma_5"] = close.rolling(5, min_periods=5).mean()
    data["sma_20"] = close.rolling(20, min_periods=20).mean()
    data["ema_12"] = close.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close.ewm(span=26, adjust=False).mean()
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    rolling_std = close.rolling(20, min_periods=20).std()
    data["rolling_vol_20"] = data["log_return_1"].rolling(20, min_periods=20).std()
    data["bollinger_z_20"] = _safe_divide(close - data["sma_20"], 2.0 * rolling_std)

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(14, min_periods=14).mean()
    avg_loss = losses.rolling(14, min_periods=14).mean()
    relative_strength = _safe_divide(avg_gain, avg_loss)
    data["rsi_14"] = 100.0 - (100.0 / (1.0 + relative_strength))

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr_14"] = true_range.rolling(14, min_periods=14).mean()

    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std()
    data["volume_z_20"] = _safe_divide(volume - volume_mean, volume_std)

    data["price_to_sma_20"] = _safe_divide(close, data["sma_20"]) - 1.0
    data["range_fraction"] = _safe_divide(high - low, close)

    day_of_week = data[date_column].dt.dayofweek.astype(float)
    data["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 5.0)
    data["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 5.0)

    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna().reset_index(drop=True)


TECHNICAL_FEATURE_COLUMNS = [
    "Close",
    "return_1",
    "log_return_1",
    "sma_5",
    "sma_20",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "rolling_vol_20",
    "bollinger_z_20",
    "rsi_14",
    "atr_14",
    "volume_z_20",
    "price_to_sma_20",
    "range_fraction",
    "dow_sin",
    "dow_cos",
]
