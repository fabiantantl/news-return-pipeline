# Preprocess for SP500 Index Price


import pandas as pd

def preprocess_sp500(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess S&P 500 index data to match stock schema.

    Input columns:
        - Date
        - S&P500

    Output columns:
        - date
        - open
        - close
        - brand_name
        - ticker
    """

    df = df_raw.copy()

    # Validate required columns
    required_columns = ["Date", "S&P500"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Rename columns
    df = df.rename(
        columns={
            "Date": "date",
            "S&P500": "close",
        }
    )

    # Convert types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce").round(2)

    # Drop invalid rows
    df = df.dropna(subset=["date", "close"])

    # Create synthetic columns (since index has no open price)
    df["open"] = df["close"]   # acceptable approximation for index-level work

    # Add identifiers (important for consistency)
    df["brand_name"] = "S&P 500"
    df["ticker"] = "SP500"

    # Sort
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    print("S&P500 processed shape:", df.shape)
    print("Date range:", df["date"].min(), "to", df["date"].max())

    return df