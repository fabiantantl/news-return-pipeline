# Sanity check to check NAN values in news_headline data set

import pandas as pd
from news_return_pipeline.paths import PROCESSED_DATA_DIR


def main():
    news_path = PROCESSED_DATA_DIR / "news_dataset.csv"

    df = pd.read_csv(news_path)

    missing_titles = df["title"].isna().sum()

    print("Total rows:", len(df))
    print("Missing titles:", missing_titles)


if __name__ == "__main__":
    main()