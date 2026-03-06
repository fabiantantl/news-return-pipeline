# DOWNLOAD STOCKS PRICE FROM KAGGLE

from pathlib import Path
import os

import kagglehub
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATASET = "nelgiriyewithana/world-stock-prices-daily-updating"


def download_stock_prices(dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Download the Kaggle stock dataset and load it."""

    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN not found in .env")

    print("Downloading stock dataset:", dataset)

    download_path = Path(kagglehub.dataset_download(dataset))
    print("Download path:", download_path)

    csv_files = sorted(download_path.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError("No CSV files found in downloaded dataset")

    source_file = csv_files[0]
    print("Selected file:", source_file.name)

    df = pd.read_csv(source_file)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    return df

# download_stock_prices()