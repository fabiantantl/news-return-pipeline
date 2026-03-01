"""CLI entrypoint for preprocessing raw data into daily aggregates."""

from __future__ import annotations

import argparse
from pathlib import Path

from news_return_pipeline.data.load_raw import load_csv
from news_return_pipeline.data.preprocess import preprocess_raw_to_daily_agg
from news_return_pipeline.paths import data_processed_dir


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to raw CSV file.")
    parser.add_argument(
        "--output",
        default=str(data_processed_dir() / "daily_agg.csv"),
        help="Output CSV path for processed daily aggregates.",
    )
    parser.add_argument(
        "--k_forward",
        type=int,
        default=5,
        help="Number of forward days used to compute ret_k.",
    )
    return parser


def main() -> None:
    """Run preprocessing pipeline from command line."""

    parser = build_parser()
    args = parser.parse_args()

    df_raw = load_csv(args.input)
    df_processed = preprocess_raw_to_daily_agg(df_raw=df_raw, k_forward=args.k_forward)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)

    min_date = df_processed["date"].min()
    max_date = df_processed["date"].max()
    print(
        "Preprocess complete | "
        f"raw_rows={len(df_raw)} | processed_days={len(df_processed)} | "
        f"date_range=[{min_date} -> {max_date}] | k_forward={args.k_forward}"
    )


if __name__ == "__main__":
    main()
