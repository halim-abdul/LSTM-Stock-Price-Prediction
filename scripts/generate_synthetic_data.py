from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, seed: int) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("Use at least 100 rows.")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=rows)
    drift = 0.00035
    seasonal = 0.002 * np.sin(np.arange(rows) / 22.0)
    shocks = rng.normal(0.0, 0.012, size=rows)
    close = 100.0 * np.exp(np.cumsum(drift + seasonal + shocks))
    open_price = close * (1.0 + rng.normal(0.0, 0.002, size=rows))
    high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.0, 0.006, size=rows))
    low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.0, 0.006, size=rows))
    volume = rng.integers(500_000, 5_000_000, size=rows)
    return pd.DataFrame({"Date": dates, "Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic.csv")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(args.rows, args.seed)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()
