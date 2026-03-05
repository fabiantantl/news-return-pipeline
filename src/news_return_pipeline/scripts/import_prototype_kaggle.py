"""Download and stage a cleaned Kaggle prototype dataset under data/raw/."""

from __future__ import annotations

import argparse

import kagglehub
import pandas as pd

from news_return_pipeline.datasets.import_kaggle import (
    normalize_prototype_schema,
    resolve_dataset_csv,
)
from news_return_pipeline.paths import data_raw_dir

DEFAULT_DATASET = "verracodeguacas/findkg-gdelt-based-financial-news-and-mentions"
DEFAULT_OUTPUT_FILENAME = "kaggle_findkg_news_clean.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug.")
    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help="Output filename to write under data/raw/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    download_path = kagglehub.dataset_download(args.dataset)
    source_csv = resolve_dataset_csv(download_path)

    raw_df = pd.read_csv(source_csv)
    cleaned_df = normalize_prototype_schema(raw_df)

    output_path = data_raw_dir() / args.output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    print(
        "Prototype dataset imported | "
        f"dataset={args.dataset} | source={source_csv} | output={output_path} | rows={len(cleaned_df)}"
    )


if __name__ == "__main__":
    main()
