# Downloading the Dataset
# ONLY FOR INSPECTION!!! DON'T MODIFY THE DATASET HERE
# KEEP IT IMMUTABLE

from __future__ import annotations

from pathlib import Path
import os

import kagglehub
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Download the FinDKG dataset from Kaggle
DEFAULT_DATASET = "verracodeguacas/findkg-gdelt-based-financial-news-and-mentions"


def download_news_headlines(
    dataset: str = DEFAULT_DATASET,
    year: int | None = None,
) -> pd.DataFrame:
    """
    Download the Kaggle dataset and load one articles parquet file.

    If year is provided, load articles_{year}.parquet.
    If year is None, load all articles_*.parquet files and concatenate them.
    """

    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN not found in .env")

    try:
        print("Token found:", True)
        print("Downloading dataset:", dataset)

        download_path = Path(kagglehub.dataset_download(dataset))
        print("Download path:", download_path)

        parquet_files = sorted(download_path.rglob("articles_*.parquet"))
        if not parquet_files:
            raise RuntimeError("No articles_*.parquet file found in downloaded dataset")

        if year is not None:
            target_name = f"articles_{year}.parquet"
            matching_files = [p for p in parquet_files if p.name == target_name]

            if not matching_files:
                available = [p.name for p in parquet_files]
                raise RuntimeError(
                    f"{target_name} not found in dataset. Available files: {available}"
                )

            source_file = matching_files[0]
            print("Selected file:", source_file.name)

            df = pd.read_parquet(source_file)
            return df

        print("No year specified. Loading all yearly parquet files...")
        print("Files found:", [p.name for p in parquet_files])

        dfs = []
        for source_file in parquet_files:
            print("Loading:", source_file.name)
            df_year = pd.read_parquet(source_file)
            dfs.append(df_year)

        df = pd.concat(dfs, ignore_index=True)
        return df

    except Exception as e:
        raise RuntimeError(f"Kaggle download/load failed: {e}") from e


if __name__ == "__main__":
    download_news_headlines(year=2020)