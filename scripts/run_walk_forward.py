from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_stock.data import clean_frame, download_market_data, load_csv
from lstm_stock.evaluate import regression_metrics
from lstm_stock.walk_forward import expanding_window_folds, summarize_fold_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanding-window persistence backtest.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--ticker")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--target", default="Close")
    parser.add_argument("--initial-train-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=60)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--output", default="artifacts/walk_forward")
    args = parser.parse_args()

    raw = load_csv(args.csv) if args.csv else download_market_data(args.ticker, args.start)
    clean = clean_frame(raw, "Date", [args.target], args.target).reset_index()
    values = clean[args.target].to_numpy(dtype=float)

    folds = expanding_window_folds(
        n_rows=len(clean),
        initial_train_size=args.initial_train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        max_folds=args.max_folds,
    )
    if not folds:
        raise ValueError("No walk-forward folds could be constructed.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    prediction_frames = []

    for fold in folds:
        test_values = values[fold.test_start:fold.test_end]
        previous = values[fold.test_start - 1:fold.test_end - 1]
        metrics = regression_metrics(test_values, previous, previous_actual=previous)
        rows.append({"fold": fold.fold, **metrics})

        prediction_frames.append(
            pd.DataFrame(
                {
                    "Date": clean["Date"].iloc[fold.test_start:fold.test_end].to_numpy(),
                    "Actual": test_values,
                    "Prediction": previous,
                    "Fold": fold.fold,
                }
            )
        )

    summary = summarize_fold_metrics(rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    summary.to_csv(output / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "folds.json").write_text(
        json.dumps([fold.__dict__ for fold in folds], indent=2),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
