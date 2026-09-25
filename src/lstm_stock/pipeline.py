from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig, validate_config
from .data import download_market_data, inverse_target, load_csv, prepare_datasets
from .evaluate import predict_scaled, regression_metrics
from .forecast import future_timestamps, recursive_forecast
from .model import LSTMRegressor, parameter_count
from .training import fit_model
from .visualize import plot_future_forecast, plot_loss, plot_test_predictions


def _run_directory(config: ExperimentConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ticker = "".join(c for c in config.data.ticker if c.isalnum() or c in "-_")
    path = Path(config.output.artifact_dir) / f"{ticker}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_experiment(config: ExperimentConfig, csv_path: str | Path | None = None) -> dict[str, object]:
    validate_config(config)
    run_dir = _run_directory(config)

    if csv_path is None:
        raw = download_market_data(config.data.ticker, config.data.start, config.data.end, config.data.interval)
        source = f"yfinance:{config.data.ticker}"
    else:
        raw = load_csv(csv_path)
        source = str(Path(csv_path))

    bundle = prepare_datasets(
        raw,
        config.data.date_column,
        config.data.features,
        config.data.target,
        config.model.sequence_length,
        config.data.train_ratio,
        config.data.val_ratio,
    )

    model = LSTMRegressor(
        input_size=len(config.data.features),
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    )

    history, device = fit_model(
        model,
        bundle.train_x,
        bundle.train_y,
        bundle.val_x,
        bundle.val_y,
        config.train,
        run_dir / "model.pt",
    )
    history.to_csv(run_dir / "training_history.csv", index=False)

    pred_scaled = predict_scaled(model, bundle.test_x, device, max(config.train.batch_size, 256))
    pred = inverse_target(pred_scaled, bundle.target_scaler)
    true = inverse_target(bundle.test_y, bundle.target_scaler)
    baseline = bundle.test_previous_raw.copy()

    lstm_metrics = regression_metrics(true, pred, bundle.test_previous_raw)
    baseline_metrics = regression_metrics(true, baseline, bundle.test_previous_raw)

    predictions = pd.DataFrame({
        "Date": bundle.test_timestamps,
        "Actual": true,
        "LSTM": pred,
        "Persistence": baseline,
        "Residual": true - pred,
    })
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    metrics = {
        "source": source,
        "device": str(device),
        "train_sequences": int(len(bundle.train_x)),
        "validation_sequences": int(len(bundle.val_x)),
        "test_sequences": int(len(bundle.test_x)),
        "trainable_parameters": int(parameter_count(model)),
        "lstm": lstm_metrics,
        "persistence_baseline": baseline_metrics,
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "resolved_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    plot_loss(history, run_dir / "loss_curve.png")
    plot_test_predictions(predictions, run_dir / "test_predictions.png")

    forecast_file = None
    if config.forecast.horizon > 0:
        last_window = bundle.full_scaled_features[-config.model.sequence_length:]
        values = recursive_forecast(
            model,
            last_window,
            config.forecast.horizon,
            bundle.target_feature_index,
            bundle.feature_scaler,
            bundle.target_scaler,
            device,
        )
        dates = future_timestamps(bundle.cleaned_frame.index[-1], config.forecast.horizon, config.data.interval)
        forecast = pd.DataFrame({"Date": dates, "Forecast": values})
        forecast_file = run_dir / "future_forecast.csv"
        forecast.to_csv(forecast_file, index=False)
        plot_future_forecast(
            bundle.cleaned_frame,
            forecast,
            config.data.target,
            run_dir / "future_forecast.png",
        )

    model.to(torch.device("cpu"))
    return {
        "artifact_dir": str(run_dir),
        "metrics": metrics,
        "forecast_file": str(forecast_file) if forecast_file else None,
    }
