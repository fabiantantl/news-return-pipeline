"""Align headline dates to the next available trading day."""

from __future__ import annotations

import pandas as pd


def align_news_to_trading_calendar(
    news_df: pd.DataFrame,
    trading_dates: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Map each headline date to the next available trading date.

    Rules:
    - If the headline date is already a trading day, keep it unchanged.
    - If the headline date is on a weekend or market holiday, map it to the
      next available trading day from the supplied trading calendar.
    - If the headline date is after the last trading date available, drop it.

    Returns a headline-level dataframe with:
    - original_date: original calendar date of the headline
    - date: mapped trading date used for aggregation/merge
    - was_mapped: whether the date changed
    - days_shifted: number of calendar days moved forward
    """

    df = news_df.copy()

    if "date" not in df.columns:
        raise ValueError("news_df must contain a 'date' column.")

    df["original_date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["original_date"].isna().any():
        invalid_count = int(df["original_date"].isna().sum())
        raise ValueError(
            f"news_df contains {invalid_count} non-parseable date value(s)."
        )

    trading_index = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(trading_dates), errors="coerce")
        .dropna()
        .dt.normalize()
        .sort_values()
        .unique()
    )

    if len(trading_index) == 0:
        raise ValueError("trading_dates is empty after parsing.")

    # searchsorted(side='left') gives:
    # - same trading day if original_date is already in the calendar
    # - next trading day if original_date is a weekend/holiday
    insertion_positions = trading_index.searchsorted(df["original_date"], side="left")

    # Drop headlines that occur after the last available trading day
    valid_mask = insertion_positions < len(trading_index)
    dropped_rows = int((~valid_mask).sum())
    if dropped_rows > 0:
        print(
            f"Dropping {dropped_rows} headline row(s) that occur after the last "
            "available trading date."
        )

    df = df.loc[valid_mask].copy()
    insertion_positions = insertion_positions[valid_mask]

    df["date"] = trading_index.take(insertion_positions).values
    df["was_mapped"] = df["date"] != df["original_date"]
    df["days_shifted"] = (df["date"] - df["original_date"]).dt.days.astype(int)

    print("Headline rows after trading-date alignment:", len(df))
    print("Mapped non-trading-day rows:", int(df["was_mapped"].sum()))
    print("Unchanged trading-day rows:", int((~df["was_mapped"]).sum()))

    return df


def build_carryover_features(mapped_news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple daily carryover features after date alignment.

    Output columns:
    - date
    - n_carryover_headlines
    - has_carryover_news
    """

    if "date" not in mapped_news_df.columns:
        raise ValueError("mapped_news_df must contain a 'date' column.")
    if "was_mapped" not in mapped_news_df.columns:
        raise ValueError("mapped_news_df must contain a 'was_mapped' column.")

    carryover_df = (
        mapped_news_df.groupby("date", as_index=False)
        .agg(
            n_carryover_headlines=("was_mapped", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    carryover_df["n_carryover_headlines"] = (
        carryover_df["n_carryover_headlines"].astype(int)
    )
    carryover_df["has_carryover_news"] = (
        carryover_df["n_carryover_headlines"] > 0
    ).astype(int)

    return carryover_df