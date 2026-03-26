# PREPROCESSING
# Raw news data
# 1) Filter data columns leaving date, title, metadata...
# 2) Normalize the date to datetime
# 3) Drop rows where the titles are NaN / empty
# 4) Stricter market-relevance filtering with OR logic
# 5) Drop duplicate headlines
# 6) Print matrix size after filtering
# 7) Renumber the rows so that order is maintained

from __future__ import annotations

import re
import pandas as pd


KEEP_COLUMNS = [
    "date",
    "title",
    "gkg_orgs",
    "gkg_themes",
    "gkg_persons",
    "gkg_dates",
]

# Main filtering signal: use more specific theme terms only
THEME_TERMS = [
    "FEDERAL_RESERVE",
    "INTEREST_RATE",
    "INTEREST_RATES",
    "INFLATION",
    "GDP",
    "UNEMPLOYMENT",
    "EARNINGS",
    "RECESSION",
    "STOCK_MARKET",
    "FINANCIAL_MARKET",
    "OIL_PRICE",
    "ENERGY",
    "COVID19",
    "COVID",
]

# High-precision macro / market title terms
TITLE_HIGH_PRECISION_TERMS = [
    r"\bs&p 500\b",
    r"\bs&p\b",
    r"\bdow jones\b",
    r"\bdow\b",
    r"\bnasdaq\b",
    r"\bwall street\b",
    r"\bfederal reserve\b",
    r"\bfed\b",
    r"\binterest rates?\b",
    r"\binflation\b",
    r"\bgdp\b",
    r"\bunemployment\b",
    r"\bjobs report\b",
    r"\bearnings\b",
    r"\brecession\b",
    r"\boil prices?\b",
    r"\bstock market\b",
    r"\bfinancial market\b",
]

# Major firm terms
TITLE_COMPANY_TERMS = [
    r"\bapple\b",
    r"\bmicrosoft\b",
    r"\bamazon\b",
    r"\btesla\b",
    r"\bgoogle\b",
    r"\balphabet\b",
    r"\bmeta\b",
    r"\bnvidia\b",
]

# Require company headlines to also mention a market / financial context
TITLE_COMPANY_CONTEXT_TERMS = [
    r"\bstock\b",
    r"\bstocks\b",
    r"\bshares\b",
    r"\bearnings\b",
    r"\brevenue\b",
    r"\bprofit\b",
    r"\bguidance\b",
    r"\bforecast\b",
    r"\bmarket\b",
    r"\bmarkets\b",
    r"\bresults\b",
    r"\bquarter\b",
    r"\bq[1-4]\b",
]

TITLE_HIGH_PRECISION_PATTERN = re.compile(
    "|".join(TITLE_HIGH_PRECISION_TERMS),
    flags=re.IGNORECASE,
)

TITLE_COMPANY_PATTERN = re.compile(
    "|".join(TITLE_COMPANY_TERMS),
    flags=re.IGNORECASE,
)

TITLE_COMPANY_CONTEXT_PATTERN = re.compile(
    "|".join(TITLE_COMPANY_CONTEXT_TERMS),
    flags=re.IGNORECASE,
)


def filter_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only relevant columns from the raw dataset."""

    df = df_raw.copy()

    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df.loc[:, KEEP_COLUMNS]

    print("Shape after column filtering:", df.shape)
    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date column to normalized datetime."""

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    return df


def _theme_match(text: str) -> bool:
    """Return True if gkg_themes contains a specific market-relevant theme."""

    if not text:
        return False

    text = str(text).upper()
    return any(term in text for term in THEME_TERMS)


def _title_match(text: str) -> bool:
    """
    Return True if title is market-relevant.

    A title is kept if:
    - it matches a high-precision macro / market term, OR
    - it mentions a major company AND also contains a financial context term
    """

    if not text:
        return False

    text = str(text)

    high_precision_match = bool(TITLE_HIGH_PRECISION_PATTERN.search(text))
    company_match = bool(TITLE_COMPANY_PATTERN.search(text))
    company_context_match = bool(TITLE_COMPANY_CONTEXT_PATTERN.search(text))

    return high_precision_match or (company_match and company_context_match)


def filter_market_relevant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a row if either:
    - gkg_themes matches specific market/economy terms, OR
    - title matches stricter market-relevant rules
    """

    df = df.copy()

    theme_mask = df["gkg_themes"].fillna("").apply(_theme_match)
    title_mask = df["title"].fillna("").apply(_title_match)
    keep_mask = theme_mask | title_mask

    filtered_df = df.loc[keep_mask].copy()

    print("Rows kept by theme filter:", int(theme_mask.sum()))
    print("Rows kept by title filter:", int(title_mask.sum()))
    print("Rows kept by OR filter:", int(keep_mask.sum()))
    print(f"Filtered matrix size: {filtered_df.shape[0]} rows x {filtered_df.shape[1]} columns")

    return filtered_df


def deduplicate_headlines(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate headlines within the same day."""

    df = df.copy()

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["date", "title"]).reset_index(drop=True)

    print("Rows after deduplication:", len(df))
    print("Dropped duplicate rows:", before_dedup - len(df))

    return df


def preprocess_news_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing pipeline for news data."""

    df = filter_columns(df_raw)
    df = normalize_dates(df)

    # clean titles and drop invalid ones
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df = df[df["title"] != ""]

    print("Rows after dropping missing titles:", len(df))

    # stricter filtering
    df = filter_market_relevant(df)

    # drop duplicate headlines before FinBERT
    df = deduplicate_headlines(df)

    # renumber rows after dropping/filtering
    df = df.reset_index(drop=True)

    print(f"Final preprocessed matrix size: {df.shape[0]} rows x {df.shape[1]} columns")
    return df