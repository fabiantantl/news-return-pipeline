from pathlib import Path
import os

import kagglehub
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATASET = "andrewmvd/sp-500-stocks"


def download_sp500_index(dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Download S&P 500 dataset and load index file."""

    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN not found in .env")

    print("Downloading S&P 500 dataset:", dataset)

    download_path = Path(kagglehub.dataset_download(dataset))
    print("Download path:", download_path)

    index_files = list(download_path.rglob("sp500_index.csv"))
    if not index_files:
        raise RuntimeError("sp500_index.csv not found in dataset")

    source_file = index_files[0]
    print("Selected file:", source_file.name)

    df = pd.read_csv(source_file)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    return df