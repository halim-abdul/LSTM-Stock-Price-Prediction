from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def expanding_window_folds(
    n_rows: int,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
    max_folds: int | None = None,
) -> list[Fold]:
    if initial_train_size < 2:
        raise ValueError("initial_train_size must be at least 2.")
    if test_size < 1:
        raise ValueError("test_size must be positive.")
    step = test_size if step_size is None else step_size
    if step < 1:
        raise ValueError("step_size must be positive.")

    folds: list[Fold] = []
    fold_index = 0
    train_end = initial_train_size

    while train_end < n_rows:
        test_start = train_end
        test_end = min(test_start + test_size, n_rows)
        if test_start >= test_end:
            break

        folds.append(
            Fold(
                fold=fold_index,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_index += 1
        if max_folds is not None and fold_index >= max_folds:
            break
        train_end += step

    return folds


def summarize_fold_metrics(rows: list[dict[str, float]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    metric_columns = [c for c in frame.columns if c != "fold"]
    summary = {
        "fold": "mean",
        **{column: float(frame[column].mean()) for column in metric_columns},
    }
    std_row = {
        "fold": "std",
        **{column: float(frame[column].std(ddof=0)) for column in metric_columns},
    }
    return pd.concat([frame, pd.DataFrame([summary, std_row])], ignore_index=True)


def persistence_predictions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) < 2:
        raise ValueError("At least two values are required.")
    return values[:-1].copy()
