# Orchestrate:
# 1) download/load raw data from Kaggle
# 2) preprocess datasets
# 3) save output to data/processed/...

import pandas as pd

from news_return_pipeline.datasets.kaggle_news import download_news_headlines
from news_return_pipeline.datasets.kaggle_stocks import download_stock_prices
from news_return_pipeline.datasets.kaggle_sp500 import download_sp500_index
from news_return_pipeline.paths import (
    data_raw_dir,
    data_processed_dir,
    get_raw_path,
    get_processed_path,
)
from news_return_pipeline.pipeline.aggregate_news import build_daily_news_features
from news_return_pipeline.pipeline.preprocess_news import preprocess_news_dataframe
from news_return_pipeline.pipeline.preprocess_sp500 import preprocess_sp500
from news_return_pipeline.pipeline.preprocess_stocks import preprocess_stocks_dataframe


def main(force_download: bool = False, run_finbert: bool = False) -> None:
    """Download raw data if needed, preprocess datasets, and save outputs."""

    data_raw_dir().mkdir(parents=True, exist_ok=True)
    data_processed_dir().mkdir(parents=True, exist_ok=True)

    news_raw_path = get_raw_path("news_dataset_raw.csv")
    news_output_path = get_processed_path("news_dataset.csv")
    news_finbert_output_path = get_processed_path("news_dataset_finbert_3m.csv")
    daily_news_output_path = get_processed_path("daily_news_features_3m.csv")

    stocks_raw_path = get_raw_path("stocks_dataset_raw.csv")
    stocks_output_path = get_processed_path("stocks_dataset.csv")

    sp500_raw_path = get_raw_path("sp500_index_raw.csv")
    sp500_output_path = get_processed_path("sp500_index.csv")

    # -------------------------
    # News raw dataset
    # -------------------------
    if news_raw_path.exists() and not force_download:
        print("Loading existing raw news dataset...")
        news_raw_df = pd.read_csv(news_raw_path)
    else:
        print("Downloading raw news dataset...")
        news_raw_df = download_news_headlines()
        news_raw_df.to_csv(news_raw_path, index=False)

    # -------------------------
    # News processed dataset
    # -------------------------
    print("Running news preprocessing from raw dataset...")
    news_df = preprocess_news_dataframe(news_raw_df)
    news_df.to_csv(news_output_path, index=False)
    print(f"Saved processed news dataset to {news_output_path}")

    # -------------------------
    # Cached FinBERT -> daily aggregated news
    # -------------------------
    if run_finbert:
        print("Using existing FinBERT output (skipping recomputation)...")

        if not news_finbert_output_path.exists():
            raise FileNotFoundError(
                f"Expected FinBERT file not found at {news_finbert_output_path}. "
                "Run FinBERT once first."
            )

        print("Loading cached FinBERT dataset...")
        news_finbert_df = pd.read_csv(news_finbert_output_path)

        print("Aggregating daily news features...")
        daily_news_df = build_daily_news_features(news_finbert_df)
        daily_news_df.to_csv(daily_news_output_path, index=False)

        print(f"Saved daily aggregated news output to {daily_news_output_path}")
    else:
        print("Skipping daily aggregation from cached FinBERT file...")

    # -------------------------
    # Multi-stock raw dataset
    # -------------------------
    if stocks_raw_path.exists() and not force_download:
        print("Loading existing raw stock dataset...")
        stocks_raw_df = pd.read_csv(stocks_raw_path)
    else:
        print("Downloading raw stock dataset...")
        stocks_raw_df = download_stock_prices()
        stocks_raw_df.to_csv(stocks_raw_path, index=False)

    # -------------------------
    # Multi-stock processed dataset
    # -------------------------
    print("Running stock preprocessing from raw dataset...")
    stocks_df = preprocess_stocks_dataframe(stocks_raw_df)
    stocks_df.to_csv(stocks_output_path, index=False)
    print(f"Saved processed stock dataset to {stocks_output_path}")

    # -------------------------
    # S&P 500 raw dataset
    # -------------------------
    if sp500_raw_path.exists() and not force_download:
        print("Loading existing raw S&P 500 index dataset...")
        sp500_raw_df = pd.read_csv(sp500_raw_path)
    else:
        print("Downloading raw S&P 500 index dataset...")
        sp500_raw_df = download_sp500_index()
        sp500_raw_df.to_csv(sp500_raw_path, index=False)

    # -------------------------
    # S&P 500 processed dataset
    # -------------------------
    print("Running S&P 500 preprocessing from raw dataset...")
    sp500_df = preprocess_sp500(sp500_raw_df)
    sp500_df.to_csv(sp500_output_path, index=False)
    print(f"Saved processed S&P 500 dataset to {sp500_output_path}")


if __name__ == "__main__":
    main(force_download=False, run_finbert=False)