# PREPROCESSING
# Raw news data
# 1) Filter Data Columns leaving data, title, meta_data...
# 2) Normalize the date to date time
# 3) Drop rows where the titles are NAN
# 4) Renumber the rows so that order is maintained

import pandas as pd


KEEP_COLUMNS = [
    "date",
    "title",
    "gkg_orgs",
    "gkg_themes",
    "gkg_persons",
    "gkg_dates",
]


def filter_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only relevant columns from the raw dataset."""

    df = df_raw.copy()

    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df.loc[:, KEEP_COLUMNS]

    print("Filtered shape:", df.shape)
    print(df.head())

    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date column to normalized datetime."""

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    return df


def preprocess_news_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing pipeline for news data."""

    df = filter_columns(df_raw)
    df = normalize_dates(df)

    # drop rows with missing titles
    df = df.dropna(subset=["title"])

    print("Rows after dropping missing titles:", len(df))
    # Drop = True
    # Renumber the rows so that order is maintained
    return df.reset_index(drop=True)