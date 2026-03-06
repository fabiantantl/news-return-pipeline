# Orchestrate:
# 1) download/load raw data from kaggle.py
# 2) preprocess via preprocess.py
# 3) save output to data/processed/...

from news_return_pipeline.datasets.kaggle import download_kaggle_dataset

from news_return_pipeline.paths import PROCESSED_DATA_DIR
from news_return_pipeline.pipeline.preprocess import preprocess_news_dataframe


def main() -> None:
    """Download, normalize, preprocess, and save the dataset."""

    df = download_kaggle_dataset()
    df = preprocess_news_dataframe(df)

    output_path = PROCESSED_DATA_DIR / "dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset built at {output_path} with {len(df)} rows")


if __name__ == "__main__":
    main()