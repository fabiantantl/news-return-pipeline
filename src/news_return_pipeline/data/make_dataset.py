"""Temporal dataset splitting utilities."""

import pandas as pd


YearRange = tuple[int, int]


def _validate_year_range(name: str, year_range: YearRange) -> None:
    start, end = year_range
    if start > end:
        raise ValueError(f"{name} year range is invalid: start ({start}) > end ({end}).")


def _ranges_overlap(a: YearRange, b: YearRange) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _filter_year_range(df: pd.DataFrame, year_range: YearRange) -> pd.DataFrame:
    start, end = year_range
    return df.loc[(df["year"] >= start) & (df["year"] <= end)].copy()


def split_by_year(
    df: pd.DataFrame,
    train_years: YearRange,
    val_years: YearRange,
    test_years: YearRange,
    min_rows_per_split: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by inclusive year ranges with strict validation rules."""

    required_columns = {"date", "year"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataframe missing required columns for splitting: {sorted(missing)}")

    for name, year_range in (
        ("train", train_years),
        ("val", val_years),
        ("test", test_years),
    ):
        _validate_year_range(name, year_range)

    if _ranges_overlap(train_years, val_years) or _ranges_overlap(train_years, test_years) or _ranges_overlap(val_years, test_years):
        raise ValueError(
            "Year ranges must be non-overlapping. "
            f"Got train={train_years}, val={val_years}, test={test_years}."
        )

    df_local = df.copy()
    df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce")
    if df_local["date"].isna().any():
        raise ValueError("Column 'date' contains non-parseable values.")

    train_df = _filter_year_range(df_local, train_years)
    val_df = _filter_year_range(df_local, val_years)
    test_df = _filter_year_range(df_local, test_years)

    for name, split_df, year_range in (
        ("train", train_df, train_years),
        ("val", val_df, val_years),
        ("test", test_df, test_years),
    ):
        if split_df.empty:
            raise ValueError(
                f"Requested {name} year range {year_range} has no rows in dataframe coverage."
            )
        if len(split_df) < min_rows_per_split:
            raise ValueError(
                f"{name} split has {len(split_df)} rows, below min_rows_per_split={min_rows_per_split}."
            )

    train_max = train_df["date"].max()
    val_min, val_max = val_df["date"].min(), val_df["date"].max()
    test_min = test_df["date"].min()

    if not (train_max < val_min):
        raise ValueError(
            "Temporal order violation: max(train.date) must be < min(val.date). "
            f"Got max(train.date)={train_max}, min(val.date)={val_min}."
        )
    if not (val_max < test_min):
        raise ValueError(
            "Temporal order violation: max(val.date) must be < min(test.date). "
            f"Got max(val.date)={val_max}, min(test.date)={test_min}."
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
