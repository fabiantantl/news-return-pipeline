# Orchestrate:
# 1) download/load raw data from kaggle.py
# 2) preprocess via preprocess.py
# 3) save output to data/processed/...

# Feature 1
# Boolean Flag to redownload the datasets from Kaggle

import pandas as pd

from news_return_pipeline.datasets.kaggle_news import download_news_headlines
from news_return_pipeline.datasets.kaggle_stocks import download_stock_prices
from news_return_pipeline.paths import PROCESSED_DATA_DIR
from news_return_pipeline.pipeline.finbert_sentiment import compute_finbert_sentiment
from news_return_pipeline.pipeline.preprocess_news import preprocess_news_dataframe
from news_return_pipeline.pipeline.preprocess_stocks import preprocess_stocks_dataframe


def main(force_download: bool = False) -> None:
    """Download, preprocess, enrich, and save the news and stock datasets."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    news_output_path = PROCESSED_DATA_DIR / "news_dataset.csv"
    news_finbert_output_path = PROCESSED_DATA_DIR / "news_dataset_finbert.csv"
    stocks_output_path = PROCESSED_DATA_DIR / "stocks_dataset.csv"

    # -------------------------
    # News dataset (base)
    # -------------------------
    if news_output_path.exists() and not force_download:
        print("Loading existing news dataset...")
        news_df = pd.read_csv(news_output_path)
    else:
        print("Downloading and preprocessing news dataset...")
        news_df = download_news_headlines()
        news_df = preprocess_news_dataframe(news_df)
        news_df.to_csv(news_output_path, index=False)

    # -------------------------
    # News dataset + FinBERT
    # -------------------------
    if news_finbert_output_path.exists() and not force_download:
        print("FinBERT dataset already exists, skipping...")
    else:
        print("Running FinBERT sentiment...")
        news_finbert_df = compute_finbert_sentiment(news_df, text_column="title")
        news_finbert_df.to_csv(news_finbert_output_path, index=False)

    # -------------------------
    # Stock dataset
    # -------------------------
    if stocks_output_path.exists() and not force_download:
        print("Loading existing stocks dataset...")
    else:
        print("Downloading and preprocessing stock dataset...")
        stocks_df = download_stock_prices()
        stocks_df = preprocess_stocks_dataframe(stocks_df)
        stocks_df.to_csv(stocks_output_path, index=False)


if __name__ == "__main__":
    # Boolean Flag to redownload the datasets from Kaggle
    # True = Redownload
    # False = Reuse
    main(force_download=True)