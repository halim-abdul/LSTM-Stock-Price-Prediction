from __future__ import annotations

import argparse
from pathlib import Path

from lstm_stock.data import download_market_data, load_csv
from lstm_stock.features import TECHNICAL_FEATURE_COLUMNS, build_technical_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe OHLCV technical-feature dataset."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Input OHLCV CSV.")
    source.add_argument("--ticker", help="Ticker to download with yfinance.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="data/technical_features.csv")
    args = parser.parse_args()

    if args.csv:
        raw = load_csv(args.csv)
    else:
        raw = download_market_data(args.ticker, args.start, args.end, "1d")

    featured = build_technical_features(raw)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output, index=False)

    print(f"Wrote {len(featured)} rows to {output}")
    print("LSTM features:")
    print(", ".join(TECHNICAL_FEATURE_COLUMNS))


if __name__ == "__main__":
    main()
