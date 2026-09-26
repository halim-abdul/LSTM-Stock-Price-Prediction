from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lstm_stock.config import load_config
from lstm_stock.data import inverse_target, load_csv, prepare_datasets
from lstm_stock.baselines import fit_random_forest_baseline, fit_ridge_baseline, score_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark non-neural baselines on LSTM sequences.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="artifacts/benchmarks.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    raw = load_csv(args.csv)
    bundle = prepare_datasets(
        raw,
        config.data.date_column,
        config.data.features,
        config.data.target,
        config.model.sequence_length,
        config.data.train_ratio,
        config.data.val_ratio,
    )

    y_true = inverse_target(bundle.test_y, bundle.target_scaler)

    ridge_scaled = fit_ridge_baseline(bundle.train_x, bundle.train_y, bundle.test_x)
    rf_scaled = fit_random_forest_baseline(bundle.train_x, bundle.train_y, bundle.test_x)

    ridge = inverse_target(ridge_scaled, bundle.target_scaler)
    forest = inverse_target(rf_scaled, bundle.target_scaler)

    results = [
        score_model("ridge", y_true, ridge),
        score_model("random_forest", y_true, forest),
        score_model("persistence", y_true, bundle.test_previous_raw),
    ]

    frame = pd.DataFrame([r.__dict__ for r in results]).sort_values("rmse")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
