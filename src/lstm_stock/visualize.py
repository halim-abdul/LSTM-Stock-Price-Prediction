from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_loss(history: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train")
    ax.plot(history["epoch"], history["val_loss"], label="Validation")
    ax.set(title="Training and validation loss", xlabel="Epoch", ylabel="Huber loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_test_predictions(predictions: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predictions["Date"], predictions["Actual"], label="Actual", linewidth=1.7)
    ax.plot(predictions["Date"], predictions["LSTM"], label="LSTM", linewidth=1.4)
    ax.plot(predictions["Date"], predictions["Persistence"], label="Persistence baseline", alpha=0.75)
    ax.set(title="Chronological test-set predictions", xlabel="Date", ylabel="Price")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_future_forecast(history: pd.DataFrame, forecast: pd.DataFrame, target: str, output_path: str | Path, history_points: int = 180) -> None:
    recent = history.tail(history_points)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(recent.index, recent[target], label="Observed", linewidth=1.7)
    ax.plot(forecast["Date"], forecast["Forecast"], label="Recursive forecast", linewidth=1.6)
    ax.axvline(recent.index[-1], linestyle="--", alpha=0.5, label="Forecast origin")
    ax.set(title="Observed history and recursive future forecast", xlabel="Date", ylabel=target)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
