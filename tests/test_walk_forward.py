import pandas as pd

from lstm_stock.walk_forward import expanding_window_folds, summarize_fold_metrics


def test_expanding_window_folds_are_ordered():
    folds = expanding_window_folds(
        n_rows=100,
        initial_train_size=40,
        test_size=10,
        step_size=10,
    )
    assert len(folds) == 6
    assert folds[0].train_end == 40
    assert folds[0].test_start == 40
    assert folds[-1].test_end == 100
    assert all(a.test_end <= b.test_start for a, b in zip(folds, folds[1:]))


def test_metric_summary_adds_mean_and_std():
    result = summarize_fold_metrics(
        [
            {"fold": 0, "mae": 1.0, "rmse": 2.0},
            {"fold": 1, "mae": 3.0, "rmse": 4.0},
        ]
    )
    assert isinstance(result, pd.DataFrame)
    assert result.iloc[-2]["fold"] == "mean"
    assert result.iloc[-1]["fold"] == "std"
    assert result.iloc[-2]["mae"] == 2.0
