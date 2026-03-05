"""CLI entrypoint for preprocessing raw data into daily aggregates."""

from __future__ import annotations

import argparse

from news_return_pipeline.config import Config
from news_return_pipeline.datasets.dataset import load_csv
from news_return_pipeline.datasets.preprocess import preprocess_raw_to_daily_agg
from news_return_pipeline.paths import get_processed_path, get_raw_path


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k_forward",
        type=int,
        default=None,
        help="Optional override for forward days used to compute ret_k.",
    )
    return parser


def main() -> None:
    """Run preprocessing pipeline from command line."""

    parser = build_parser()
    args = parser.parse_args()

    config = Config()
    k_forward = config.k_forward if args.k_forward is None else args.k_forward
    input_path = get_raw_path(config.raw_filename)
    output_path = get_processed_path(config.processed_filename)

    df_raw = load_csv(input_path)
    df_processed = preprocess_raw_to_daily_agg(df_raw=df_raw, k_forward=k_forward)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)

    min_date = df_processed["date"].min()
    max_date = df_processed["date"].max()
    print(
        "Preprocess complete | "
        f"input={input_path} | output={output_path} | "
        f"raw_rows={len(df_raw)} | processed_days={len(df_processed)} | "
        f"date_range=[{min_date} -> {max_date}] | k_forward={k_forward}"
    )


if __name__ == "__main__":
    main()
