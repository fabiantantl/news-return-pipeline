"""Utilities for importing a cleaned prototyping dataset from Kaggle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATE_CANDIDATES = ("date", "Date", "datetime", "Datetime", "published_date", "timestamp", "time")
HEADLINE_CANDIDATES = (
    "headline",
    "Headline",
    "news",
    "News",
    "title",
    "Title",
    "headlines",
)
CLOSE_CANDIDATES = (
    "close",
    "Close",
    "adj_close",
    "Adj Close",
    "sp500_close",
    "S&P500_Close",
    "close_price",
    "price",
    "Price",
    "stock_close",
)


def _pick_column(columns: list[str], candidates: tuple[str, ...], semantic_name: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not find a '{semantic_name}' column. Available columns: {columns}"
    )


def normalize_prototype_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Kaggle dataframe into required pipeline schema.

    Returns a dataframe with exactly: ``date``, ``headline``, ``close``.
    """

    columns = list(df.columns)
    date_col = _pick_column(columns, DATE_CANDIDATES, "date")
    headline_col = _pick_column(columns, HEADLINE_CANDIDATES, "headline")
    close_col = _pick_column(columns, CLOSE_CANDIDATES, "close")

    cleaned = df.loc[:, [date_col, headline_col, close_col]].copy()
    cleaned.columns = ["date", "headline", "close"]

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["headline"] = cleaned["headline"].astype(str).str.strip()
    cleaned["close"] = pd.to_numeric(cleaned["close"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date", "headline", "close"]).reset_index(drop=True)

    return cleaned


def resolve_dataset_csv(download_dir: str | Path) -> Path:
    """Pick the most likely CSV file from a KaggleHub dataset download directory."""

    root = Path(download_dir)
    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {root}")
    return csv_files[0]
