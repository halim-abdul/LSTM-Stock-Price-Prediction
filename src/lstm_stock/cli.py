from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import load_config, validate_config
from .pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, evaluate, and forecast with an LSTM stock model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--csv", default=None, help="Optional local CSV instead of yfinance.")
    parser.add_argument("--ticker", default=None, help="Optional ticker override.")
    parser.add_argument("--horizon", type=int, default=None, help="Optional forecast horizon override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.ticker is not None:
        config.data = replace(config.data, ticker=args.ticker)
    if args.horizon is not None:
        config.forecast = replace(config.forecast, horizon=args.horizon)

    validate_config(config)
    print(json.dumps(run_experiment(config, args.csv), indent=2))


if __name__ == "__main__":
    main()
