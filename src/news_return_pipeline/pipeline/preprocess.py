"""Preprocessing functions for raw headline/price data."""

import pandas as pd

from news_return_pipeline.config import Config


def aggregate_headlines(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw rows into one row per day."""

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    grouped = (
        df.groupby("date", sort=False)
        .agg(
            text=("headline", lambda values: " [SEP] ".join(values.astype(str))),
            n_headlines=("headline", "size"),
            close=("close", "last"),
        )
        .reset_index()
    )

    grouped["year"] = grouped["date"].dt.year
    grouped = grouped.sort_values("date", ascending=True).reset_index(drop=True)
    return grouped


def add_forward_return(df_daily: pd.DataFrame, k_forward: int) -> pd.DataFrame:
    """Add k-forward return label ``ret_k`` and drop rows without future labels."""

    if k_forward <= 0:
        raise ValueError(f"k_forward must be positive; got {k_forward}.")

    df = df_daily.copy()
    df["ret_k"] = df["close"].shift(-k_forward) / df["close"] - 1.0
    df = df.iloc[:-k_forward]
    df = df.dropna(subset=["ret_k"]).reset_index(drop=True)
    return df


def preprocess_news_dataframe(df_raw: pd.DataFrame, k_forward: int | None = None) -> pd.DataFrame:
    """Convert raw rows into a daily aggregated dataset with forward returns."""

    effective_k = Config().k_forward if k_forward is None else k_forward
    df_daily = aggregate_headlines(df_raw)
    df_processed = add_forward_return(df_daily, k_forward=effective_k)
    expected_order = ["date", "year", "text", "n_headlines", "close", "ret_k"]
    return df_processed.loc[:, expected_order]
