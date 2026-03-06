from pathlib import Path

import pandas as pd

from news_return_pipeline.datasets.kaggle_news import normalize_prototype_schema


def test_normalize_prototype_schema_fixture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "prototype.csv"
    raw_df = pd.read_csv(fixture)

    result = normalize_prototype_schema(raw_df)

    assert list(result.columns) == ["date", "headline", "close"]
    assert result["date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert result["headline"].tolist() == ["Markets rise", "Fed signals pause"]
    assert result["close"].tolist() == [4800.5, 4821.0]

from news_return_pipeline.datasets.kaggle_news import download_kaggle_dataset

df = download_kaggle_dataset()

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nPreview:")
print(df.head(10))


# if __name__ == "__main__":
#     test_normalize_prototype_schema_fixture()
#     print("Test passed")