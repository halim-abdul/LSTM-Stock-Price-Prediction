import numpy as np
import pandas as pd

from lstm_stock.data import prepare_datasets


def test_prepare_datasets_is_chronological_and_train_scaled_only():
    rows = 200
    frame = pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=rows),
        "Close": np.linspace(10.0, 50.0, rows),
    })
    bundle = prepare_datasets(frame, "Date", ["Close"], "Close", 20, 0.70, 0.15)
    train_end = int(rows * 0.70)
    assert bundle.train_x.shape[1:] == (20, 1)
    assert bundle.val_x.shape[1:] == (20, 1)
    assert bundle.test_x.shape[1:] == (20, 1)
    assert np.isclose(bundle.feature_scaler.mean_[0], frame["Close"].iloc[:train_end].mean())
    assert bundle.test_timestamps.is_monotonic_increasing
