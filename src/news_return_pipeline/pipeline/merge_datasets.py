"""Merge daily news features with S&P 500 prices."""

from __future__ import annotations

import pandas as pd


def merge_news_with_sp500(
    news_df: pd.DataFrame,
    sp500_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily news features with S&P 500 prices on date.

    Keeps only dates that exist in both datasets.
    """

    news = news_df.copy()
    sp500 = sp500_df.copy()

    if "date" not in news.columns:
        raise ValueError("news_df must contain a 'date' column.")
    if "date" not in sp500.columns:
        raise ValueError("sp500_df must contain a 'date' column.")

    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.normalize()
    sp500["date"] = pd.to_datetime(sp500["date"], errors="coerce").dt.normalize()

    if news["date"].isna().any():
        raise ValueError("news_df contains non-parseable dates.")
    if sp500["date"].isna().any():
        raise ValueError("sp500_df contains non-parseable dates.")

    merged_df = (
        news.merge(sp500, on="date", how="inner", suffixes=("", "_sp500"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    if merged_df.empty:
        raise ValueError("Merged dataframe is empty. Check date overlap between datasets.")

    print("Merged dataframe shape:", merged_df.shape)
    print("Merged date range:", merged_df["date"].min(), "to", merged_df["date"].max())

    return merged_df