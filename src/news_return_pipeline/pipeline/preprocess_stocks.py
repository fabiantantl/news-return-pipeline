# PREPROCESS STOCK PRICES
# Cleaning up the data...
# 1) ignore timezone differences as we are working on a 
# time regime of 5 days so the fidelity is not neccessary
# 2) Clean up stock prices from floating precision to 2 d.p (Standard for Stock Prices)
# 3) Schema Normalization


import pandas as pd


KEEP_COLUMNS = [
    "Date",
    "Open",
    "Close",
    "Brand_Name",
    "Ticker",
]


def preprocess_stocks_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns, normalize schema, and standardize dates."""

    df = df_raw.copy()

    # validate expected columns
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # keep and rename columns to match project schema
    df = (
        df.loc[:, KEEP_COLUMNS]
        .rename(
            columns={
                "Date": "date",
                "Open": "open",
                "Close": "close",
                "Brand_Name": "brand_name",
                "Ticker": "ticker",
            }
        )
    )

    # Clean floating precision
    df["open"] = pd.to_numeric(df["open"]).round(2)
    df["close"] = pd.to_numeric(df["close"]).round(2)

    # normalize date format (remove timezone + truncate to day)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()

    # ensure proper ordering for time series operations
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # print("Shape:", df.shape)
    # print("Columns:", df.columns.tolist())
    # print(df.head())

    return df