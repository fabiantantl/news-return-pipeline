# Orchestrate:
# 1) download/load raw data from kaggle.py
# 2) preprocess via preprocess.py
# 3) save output to data/processed/...

# Feature 1
# Boolean Flag to redownload the datasets from Kaggle

import pandas as pd

from news_return_pipeline.datasets.kaggle_news import download_news_headlines
from news_return_pipeline.datasets.kaggle_stocks import download_stock_prices
from news_return_pipeline.paths import (
    data_raw_dir,
    data_processed_dir,
    get_raw_path,
    get_processed_path,
)
from news_return_pipeline.pipeline.finbert_sentiment import compute_finbert_sentiment
from news_return_pipeline.pipeline.preprocess_news import preprocess_news_dataframe
from news_return_pipeline.pipeline.preprocess_stocks import preprocess_stocks_dataframe


def main(force_download: bool = False) -> None:
    """Download raw data if needed, preprocess from raw, and save outputs."""

    data_raw_dir().mkdir(parents=True, exist_ok=True)
    data_processed_dir().mkdir(parents=True, exist_ok=True)

    news_raw_path = get_raw_path("news_dataset_raw.csv")
    news_output_path = get_processed_path("news_dataset.csv")
    news_finbert_output_path = get_processed_path("news_dataset_finbert.csv")

    stocks_raw_path = get_raw_path("stocks_dataset_raw.csv")
    stocks_output_path = get_processed_path("stocks_dataset.csv")

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
    print("Running preprocessing from raw dataset...")
    news_df = preprocess_news_dataframe(news_raw_df)
    news_df.to_csv(news_output_path, index=False)

    # -------------------------
    # News dataset + FinBERT
    # -------------------------
    if news_finbert_output_path.exists() and not force_download:
        print("FinBERT dataset already exists, skipping...")
    else:
        print("Running FinBERT sentiment...")

        # Use a small subset for initial testing
        news_df_small = news_df.head(500).copy()

        news_finbert_df = compute_finbert_sentiment(
            news_df_small,
            text_column="title",
            batch_size=32,
        )
        news_finbert_df.to_csv(news_finbert_output_path, index=False)

    # -------------------------
    # Stock raw dataset
    # -------------------------
    if stocks_raw_path.exists() and not force_download:
        print("Loading existing raw stock dataset...")
        stocks_raw_df = pd.read_csv(stocks_raw_path)
    else:
        print("Downloading raw stock dataset...")
        stocks_raw_df = download_stock_prices()
        stocks_raw_df.to_csv(stocks_raw_path, index=False)

    # -------------------------
    # Stock processed dataset
    # -------------------------
    print("Running stock preprocessing from raw dataset...")
    stocks_df = preprocess_stocks_dataframe(stocks_raw_df)
    stocks_df.to_csv(stocks_output_path, index=False)


if __name__ == "__main__":
    main(force_download=False)