"""Aggregate headline-level news sentiment into daily features."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["date", "title", "sentiment_label", "sentiment_score"]


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that the dataframe contains the required columns."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for news aggregation: {missing}")


def add_signed_sentiment(
    df: pd.DataFrame,
    label_column: str = "sentiment_label",
    score_column: str = "sentiment_score",
    output_column: str = "sentiment_signed",
) -> pd.DataFrame:
    """
    Convert FinBERT label + confidence into a signed scalar score.

    Baseline encoding:
    - positive -> +score
    - negative -> -score
    - neutral  -> 0.0
    """
    df_local = df.copy()

    if label_column not in df_local.columns:
        raise ValueError(f"Column '{label_column}' not found in dataframe.")
    if score_column not in df_local.columns:
        raise ValueError(f"Column '{score_column}' not found in dataframe.")

    df_local[label_column] = df_local[label_column].astype(str).str.lower().str.strip()
    df_local[score_column] = pd.to_numeric(df_local[score_column], errors="coerce")

    if df_local[score_column].isna().any():
        invalid_count = int(df_local[score_column].isna().sum())
        raise ValueError(
            f"Column '{score_column}' contains {invalid_count} non-numeric value(s)."
        )

    valid_labels = {"positive", "negative", "neutral"}
    invalid_labels = sorted(set(df_local[label_column]) - valid_labels)
    if invalid_labels:
        raise ValueError(
            f"Unexpected sentiment label(s): {invalid_labels}. "
            f"Expected one of {sorted(valid_labels)}."
        )

    def map_signed_score(row: pd.Series) -> float:
        label = row[label_column]
        score = row[score_column]

        if label == "positive":
            return float(score)
        if label == "negative":
            return -float(score)
        return 0.0

    df_local[output_column] = df_local.apply(map_signed_score, axis=1)
    return df_local


def aggregate_daily_news(
    df: pd.DataFrame,
    date_column: str = "date",
    title_column: str = "title",
    signed_column: str = "sentiment_signed",
    join_text: bool = True,
    text_separator: str = " [SEP] ",
) -> pd.DataFrame:
    """
    Aggregate headline-level news into one row per day.
    """
    df_local = df.copy()

    required = [date_column, title_column, "sentiment_label", signed_column]
    missing = [column for column in required if column not in df_local.columns]
    if missing:
        raise ValueError(f"Missing required columns for daily aggregation: {missing}")

    df_local[date_column] = pd.to_datetime(
        df_local[date_column], errors="coerce"
    ).dt.normalize()

    if df_local[date_column].isna().any():
        invalid_count = int(df_local[date_column].isna().sum())
        raise ValueError(
            f"Column '{date_column}' contains {invalid_count} non-parseable date value(s)."
        )

    df_local[title_column] = df_local[title_column].fillna("").astype(str).str.strip()
    df_local = df_local[df_local[title_column] != ""].copy()

    df_local["sentiment_label"] = (
        df_local["sentiment_label"].astype(str).str.lower().str.strip()
    )

    daily = (
        df_local.groupby(date_column, as_index=False)
        .agg(
            sentiment_mean=(signed_column, "mean"),
            sentiment_std=(signed_column, "std"),
            n_headlines=(title_column, "count"),
            n_positive=("sentiment_label", lambda s: int((s == "positive").sum())),
            n_negative=("sentiment_label", lambda s: int((s == "negative").sum())),
            n_neutral=("sentiment_label", lambda s: int((s == "neutral").sum())),
        )
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    daily["year"] = daily[date_column].dt.year
    daily["frac_positive"] = daily["n_positive"] / daily["n_headlines"]
    daily["frac_negative"] = daily["n_negative"] / daily["n_headlines"]
    daily["frac_neutral"] = daily["n_neutral"] / daily["n_headlines"]

    if join_text:
        daily_text = (
            df_local.groupby(date_column)[title_column]
            .apply(lambda titles: text_separator.join(titles.astype(str)))
            .reset_index(name="text")
        )
        daily = daily.merge(daily_text, on=date_column, how="left")

    column_order = [
        date_column,
        "year",
        "sentiment_mean",
        "sentiment_std",
        "n_headlines",
        "n_positive",
        "n_negative",
        "n_neutral",
        "frac_positive",
        "frac_negative",
        "frac_neutral",
    ]

    if join_text:
        column_order.append("text")

    daily = daily.loc[:, column_order]

    print("Daily aggregated news shape:", daily.shape)
    print("Date range:", daily[date_column].min(), "to", daily[date_column].max())
    print("Total daily rows:", len(daily))

    return daily


def build_daily_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full headline-to-daily aggregation pipeline."""
    _validate_required_columns(df)
    df_with_signed = add_signed_sentiment(df)
    daily_df = aggregate_daily_news(df_with_signed)
    return daily_df