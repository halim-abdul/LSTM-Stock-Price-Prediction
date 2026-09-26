from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from lstm_stock.model import LSTMRegressor
from lstm_stock.uncertainty import mc_dropout_predict, prediction_interval


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate MC-dropout uncertainty.")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    raw = rng.normal(size=(16, 30, 1)).astype(np.float32)

    model = LSTMRegressor(input_size=1, hidden_size=32, num_layers=2, dropout=0.3)
    device = torch.device("cpu")
    model.to(device)

    draws = mc_dropout_predict(model, raw, device=device, samples=args.samples)
    mean, low, high = prediction_interval(draws, 0.05, 0.95)

    scaler = StandardScaler().fit(np.linspace(90.0, 110.0, 100).reshape(-1, 1))
    mean_raw = scaler.inverse_transform(mean.reshape(-1, 1)).reshape(-1)
    low_raw = scaler.inverse_transform(low.reshape(-1, 1)).reshape(-1)
    high_raw = scaler.inverse_transform(high.reshape(-1, 1)).reshape(-1)

    for i in range(min(5, len(mean_raw))):
        print(
            f"sample={i} mean={mean_raw[i]:.3f} "
            f"p05={low_raw[i]:.3f} p95={high_raw[i]:.3f}"
        )


if __name__ == "__main__":
    main()
