# Orchestrate:
# 1) download/load raw data from kaggle.py
# 2) preprocess via preprocess.py
# 3) save output to data/processed/...

from news_return_pipeline.datasets.kaggle_news import download_kaggle_dataset
from news_return_pipeline.datasets.kaggle_stocks import download_stock_prices

from news_return_pipeline.paths import PROCESSED_DATA_DIR
from news_return_pipeline.pipeline.preprocess_news import preprocess_news_dataframe
from news_return_pipeline.pipeline.preprocess_stocks import preprocess_stocks_dataframe

def main() -> None:
    """Download, preprocess, and save the news and stock datasets."""

    # News dataset
    news_df = download_kaggle_dataset()
    news_df = preprocess_news_dataframe(news_df)

    news_output_path = PROCESSED_DATA_DIR / "news_dataset.csv"
    news_output_path.parent.mkdir(parents=True, exist_ok=True)
    news_df.to_csv(news_output_path, index=False)

    print(f"News dataset built at {news_output_path} with {len(news_df)} rows")

    # Stock dataset
    stocks_df = download_stock_prices()
    stocks_df = preprocess_stocks_dataframe(stocks_df)

    stocks_output_path = PROCESSED_DATA_DIR / "stocks_dataset.csv"
    stocks_output_path.parent.mkdir(parents=True, exist_ok=True)
    stocks_df.to_csv(stocks_output_path, index=False)

    print(f"Stocks dataset built at {stocks_output_path} with {len(stocks_df)} rows")


if __name__ == "__main__":
    main()