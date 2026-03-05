"""Single-entry dataset build pipeline."""

from news_return_pipeline.datasets.kaggle import (
    download_kaggle_dataset,
    normalize_prototype_schema,
)
from news_return_pipeline.paths import PROCESSED_DATA_DIR
from news_return_pipeline.pipeline.preprocess import preprocess_news_dataframe


def main() -> None:
    """Download, normalize, preprocess, and save the dataset."""

    raw_df = download_kaggle_dataset()
    df = normalize_prototype_schema(raw_df)
    df = preprocess_news_dataframe(df)
    output_path = PROCESSED_DATA_DIR / "dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset built successfully at {output_path} with {len(df)} rows")


if __name__ == "__main__":
    main()
