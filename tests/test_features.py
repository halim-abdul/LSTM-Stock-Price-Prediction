import numpy as np
import pandas as pd

from lstm_stock.features import build_technical_features


def _sample_frame(rows: int = 80) -> pd.DataFrame:
    close = np.linspace(100.0, 130.0, rows) + np.sin(np.arange(rows) / 4.0)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-01", periods=rows),
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(1_000_000, 2_000_000, rows),
        }
    )


def test_feature_builder_returns_finite_features():
    featured = build_technical_features(_sample_frame())
    assert not featured.empty
    assert featured.isna().sum().sum() == 0
    assert np.isfinite(featured.select_dtypes(include=[np.number])).all().all()


def test_feature_builder_is_causal_for_earlier_rows():
    frame = _sample_frame()
    original = build_technical_features(frame)

    changed = frame.copy()
    changed.loc[changed.index[-1], "Close"] *= 10.0
    modified = build_technical_features(changed)

    common = min(len(original), len(modified)) - 1
    columns = ["sma_20", "ema_12", "rsi_14", "atr_14"]
    assert np.allclose(
        original.loc[: common - 1, columns],
        modified.loc[: common - 1, columns],
    )
