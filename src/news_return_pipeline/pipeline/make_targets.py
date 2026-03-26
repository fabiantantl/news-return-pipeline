"""Create forward-return targets on the merged dataframe."""

from __future__ import annotations

import pandas as pd


def add_forward_return_target(
    df: pd.DataFrame,
    horizon: int = 5,
    price_column: str = "close",
    target_column: str | None = None,
) -> pd.DataFrame:
    """
    Add forward close-to-close return target.

    target[t] = (close[t + horizon] / close[t]) - 1
    """

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")

    df_local = df.copy()

    if price_column not in df_local.columns:
        raise ValueError(f"Column '{price_column}' not found in dataframe.")

    df_local[price_column] = pd.to_numeric(df_local[price_column], errors="coerce")
    if df_local[price_column].isna().any():
        invalid_count = int(df_local[price_column].isna().sum())
        raise ValueError(
            f"Column '{price_column}' contains {invalid_count} non-numeric value(s)."
        )

    if target_column is None:
        target_column = f"target_return_{horizon}d"

    df_local = df_local.sort_values("date").reset_index(drop=True)

    future_price = df_local[price_column].shift(-horizon)
    df_local[target_column] = (future_price / df_local[price_column]) - 1.0

    print(f"Created target column: {target_column}")
    print("Rows with missing target:", int(df_local[target_column].isna().sum()))

    return df_local