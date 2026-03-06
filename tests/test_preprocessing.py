# Financial News 
# Sanity Check for NAN values in the title

import pandas as pd

from news_return_pipeline.datasets.kaggle_news import download_news_headlines
from news_return_pipeline.paths import PROCESSED_DATA_DIR
from news_return_pipeline.pipeline.preprocess_news import preprocess_news_dataframe


def main():
    """Download, preprocess, save, and run sanity check."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Download + preprocess
    # -------------------------
    print("Downloading news dataset...")
    news_df = download_news_headlines()

    print("Preprocessing news dataset...")
    news_df = preprocess_news_dataframe(news_df)

    news_output_path = PROCESSED_DATA_DIR / "news_dataset.csv"
    news_df.to_csv(news_output_path, index=False)

    print(f"Saved dataset to {news_output_path}")

    # -------------------------
    # Sanity check
    # -------------------------
    missing_titles = news_df["title"].isna().sum()

    print("Total rows:", len(news_df))
    print("Missing titles:", missing_titles)

    assert missing_titles == 0

    print("sanity check passed")


if __name__ == "__main__":
    main()