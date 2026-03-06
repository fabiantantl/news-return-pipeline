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


def download_kaggle_dataset(dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Download the Kaggle dataset and load one articles parquet file."""

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

        source_file = parquet_files[0]
        print("Selected file:", source_file.name)

        df = pd.read_parquet(source_file)

        # print("Shape:", df.shape)
        # print("Columns:", df.columns.tolist())
        # print(df.head())

        # print(df["gkg_themes"].head(20))
        # print(df["gkg_orgs"].head(20))
        # print(df["gkg_persons"].head(20))
        return df

    except Exception as e:
        raise RuntimeError(f"Kaggle download/load failed: {e}") from e


if __name__ == "__main__":
    download_kaggle_dataset()
