"""Build raw, processed, and model-ready datasets for the news-return pipeline."""

from __future__ import annotations

import pandas as pd

from news_return_pipeline.datasets.kaggle_news import download_news_headlines
from news_return_pipeline.datasets.kaggle_sp500 import download_sp500_index
from news_return_pipeline.datasets.kaggle_stocks import download_stock_prices
from news_return_pipeline.paths import (
    data_processed_dir,
    data_raw_dir,
    get_processed_path,
    get_raw_path,
)
from news_return_pipeline.pipeline.aggregate_news import build_daily_news_features
from news_return_pipeline.pipeline.align_news_to_trading_calendar import (
    align_news_to_trading_calendar,
    build_carryover_features,
)
from news_return_pipeline.pipeline.finbert_sentiment import compute_finbert_sentiment
from news_return_pipeline.pipeline.make_targets import add_forward_return_target
from news_return_pipeline.pipeline.preprocess_news import preprocess_news_dataframe
from news_return_pipeline.pipeline.preprocess_sp500 import preprocess_sp500
from news_return_pipeline.pipeline.preprocess_stocks import preprocess_stocks_dataframe


YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def build_news_raw(force_download: bool = False) -> pd.DataFrame:
    """Load combined raw news data from disk or download selected yearly files and concatenate."""

    news_raw_path = get_raw_path("news_raw.csv")
    if news_raw_path.exists() and not force_download:
        print("Loading existing combined raw news dataset...")
        return pd.read_csv(news_raw_path)

    print("Downloading yearly raw news datasets and concatenating...")
    yearly_frames = []

    for year in YEARS:
        print(f"Loading raw news for year {year}...")
        year_df = download_news_headlines(year=year)
        year_df["source_year"] = year
        yearly_frames.append(year_df)

    news_raw_df = pd.concat(yearly_frames, ignore_index=True)
    news_raw_df.to_csv(news_raw_path, index=False)
    print(f"Saved combined raw news dataset to {news_raw_path}")
    print("Combined raw news shape:", news_raw_df.shape)

    return news_raw_df


def build_news_preprocessed(news_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw news and save processed headline-level data."""

    news_preprocessed_df = preprocess_news_dataframe(news_raw_df)
    news_preprocessed_path = get_processed_path("news_preprocessed.csv")
    news_preprocessed_df.to_csv(news_preprocessed_path, index=False)
    print(f"Saved preprocessed news dataset to {news_preprocessed_path}")
    return news_preprocessed_df


def build_finbert_scored_with_cache(
    news_preprocessed_df: pd.DataFrame,
    run_finbert: bool = True,
) -> pd.DataFrame:
    """Score headlines with FinBERT using a reusable date+title cache."""

    cache_path = get_processed_path("news_finbert_scored.csv")
    key_columns = ["date", "title"]

    if cache_path.exists():
        cache_df = pd.read_csv(cache_path)
        print(f"Loaded existing FinBERT cache from {cache_path}")
    else:
        cache_df = pd.DataFrame(
            columns=list(news_preprocessed_df.columns) + [
                "sentiment_label",
                "sentiment_score",
            ]
        )
        print("No existing FinBERT cache found; creating a new one.")

    for df_name, df in (("news_preprocessed_df", news_preprocessed_df), ("cache_df", cache_df)):
        if "date" not in df.columns or "title" not in df.columns:
            raise ValueError(f"{df_name} must contain 'date' and 'title'.")

    news_local_df = news_preprocessed_df.copy()
    cache_local_df = cache_df.copy()

    news_local_df["date"] = pd.to_datetime(news_local_df["date"], errors="coerce").dt.normalize()
    cache_local_df["date"] = pd.to_datetime(cache_local_df["date"], errors="coerce").dt.normalize()

    if news_local_df["date"].isna().any():
        raise ValueError("news_preprocessed_df contains non-parseable dates.")

    cache_local_df = cache_local_df.dropna(subset=["date", "title"]).copy()
    cache_local_df = cache_local_df.drop_duplicates(subset=key_columns, keep="last")

    merged_df = news_local_df.merge(
        cache_local_df[key_columns],
        on=key_columns,
        how="left",
        indicator=True,
    )
    rows_to_score_df = merged_df.loc[merged_df["_merge"] == "left_only", news_local_df.columns]

    if len(rows_to_score_df) > 0:
        if not run_finbert:
            raise FileNotFoundError(
                "FinBERT cache is missing required rows and run_finbert is False. "
                "Rerun with run_finbert=True to score missing headlines."
            )

        print(f"Running FinBERT for {len(rows_to_score_df)} uncached headline rows...")
        new_scores_df = compute_finbert_sentiment(rows_to_score_df, text_column="title")
        cache_local_df = pd.concat([cache_local_df, new_scores_df], ignore_index=True)
    else:
        print("All rows already present in FinBERT cache; no new scoring needed.")

    cache_local_df = cache_local_df.drop_duplicates(subset=key_columns, keep="last")
    cache_local_df = cache_local_df.sort_values(key_columns).reset_index(drop=True)
    cache_local_df.to_csv(cache_path, index=False)
    print(f"Saved FinBERT cache to {cache_path}")

    news_finbert_df = news_local_df.merge(
        cache_local_df,
        on=key_columns,
        how="left",
    )

    required_sentiment_columns = ["sentiment_label", "sentiment_score"]
    missing_sentiment_columns = [
        c for c in required_sentiment_columns if c not in news_finbert_df.columns
    ]
    if missing_sentiment_columns:
        raise ValueError(
            f"Missing sentiment column(s) after cache merge: {missing_sentiment_columns}"
        )

    if news_finbert_df[required_sentiment_columns].isna().any().any():
        raise ValueError("Some rows are missing FinBERT outputs after cache merge.")

    return news_finbert_df


def build_news_aligned(
    news_finbert_df: pd.DataFrame,
    sp500_preprocessed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align headline dates to trading days and build carryover features."""

    news_aligned_df = align_news_to_trading_calendar(
        news_finbert_df,
        sp500_preprocessed_df["date"],
    )
    news_aligned_path = get_processed_path("news_aligned.csv")
    news_aligned_df.to_csv(news_aligned_path, index=False)
    print(f"Saved aligned headline dataset to {news_aligned_path}")

    news_carryover_df = build_carryover_features(news_aligned_df)
    news_carryover_path = get_processed_path("news_carryover_features.csv")
    news_carryover_df.to_csv(news_carryover_path, index=False)
    print(f"Saved carryover feature dataset to {news_carryover_path}")

    return news_aligned_df, news_carryover_df


def build_news_daily_features(
    news_aligned_df: pd.DataFrame,
    news_carryover_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate aligned/scored headlines into daily news features."""

    news_daily_df = build_daily_news_features(news_aligned_df)
    news_daily_df = (
        news_daily_df.merge(news_carryover_df, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    news_daily_df["n_carryover_headlines"] = (
        news_daily_df["n_carryover_headlines"].fillna(0).astype(int)
    )
    news_daily_df["has_carryover_news"] = (
        news_daily_df["has_carryover_news"].fillna(0).astype(int)
    )

    news_daily_path = get_processed_path("news_daily_features.csv")
    news_daily_df.to_csv(news_daily_path, index=False)
    print(f"Saved daily news feature dataset to {news_daily_path}")
    return news_daily_df


def build_stocks_preprocessed(stocks_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess multi-stock data and save cleaned output."""

    stocks_preprocessed_df = preprocess_stocks_dataframe(stocks_raw_df)
    stocks_preprocessed_path = get_processed_path("stocks_preprocessed.csv")
    stocks_preprocessed_df.to_csv(stocks_preprocessed_path, index=False)
    print(f"Saved preprocessed multi-stock dataset to {stocks_preprocessed_path}")
    return stocks_preprocessed_df


def build_sp500_raw(force_download: bool = False) -> pd.DataFrame:
    """Load raw S&P 500 data from disk or download from Kaggle."""

    sp500_raw_path = get_raw_path("sp500_raw.csv")
    if sp500_raw_path.exists() and not force_download:
        print("Loading existing raw S&P 500 dataset...")
        return pd.read_csv(sp500_raw_path)

    print("Downloading raw S&P 500 dataset...")
    sp500_raw_df = download_sp500_index()
    sp500_raw_df.to_csv(sp500_raw_path, index=False)
    print(f"Saved raw S&P 500 dataset to {sp500_raw_path}")
    return sp500_raw_df


def build_sp500_preprocessed(sp500_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess S&P 500 data and save cleaned output."""

    sp500_preprocessed_df = preprocess_sp500(sp500_raw_df)
    sp500_preprocessed_path = get_processed_path("sp500_preprocessed.csv")
    sp500_preprocessed_df.to_csv(sp500_preprocessed_path, index=False)
    print(f"Saved preprocessed S&P 500 dataset to {sp500_preprocessed_path}")
    return sp500_preprocessed_df


def build_sp500_targets(sp500_preprocessed_df: pd.DataFrame) -> pd.DataFrame:
    """Create forward-return targets and save trading-day target dataset."""

    sp500_targets_df = add_forward_return_target(sp500_preprocessed_df, horizon=5)
    sp500_targets_path = get_processed_path("sp500_with_targets.csv")
    sp500_targets_df.to_csv(sp500_targets_path, index=False)
    print(f"Saved S&P 500 targets dataset to {sp500_targets_path}")
    return sp500_targets_df


def build_model_dataset(
    news_daily_df: pd.DataFrame,
    sp500_targets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily news features with target-bearing S&P 500 rows."""

    model_df = (
        sp500_targets_df.merge(news_daily_df, on="date", how="left", suffixes=("", "_news"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    model_df["year"] = model_df["date"].dt.year

    zero_fill_columns = [
        "sentiment_mean",
        "sentiment_std",
        "n_headlines",
        "n_positive",
        "n_negative",
        "n_neutral",
        "frac_positive",
        "frac_negative",
        "frac_neutral",
        "n_carryover_headlines",
        "has_carryover_news",
    ]

    for column in zero_fill_columns:
        if column in model_df.columns:
            model_df[column] = model_df[column].fillna(0)

    if "text" in model_df.columns:
        model_df["text"] = model_df["text"].fillna("")

    model_df = model_df.dropna(subset=["target_return_5d"]).reset_index(drop=True)

    if "text" in model_df.columns:
        model_df = model_df[[c for c in model_df.columns if c != "text"] + ["text"]]

    model_path = get_processed_path("model_dataset.csv")
    model_df.to_csv(model_path, index=False)
    print(f"Saved final model dataset to {model_path}")
    return model_df


def build_stocks_raw(force_download: bool = False) -> pd.DataFrame:
    """Load raw multi-stock data from disk or download from Kaggle."""

    stocks_raw_path = get_raw_path("stocks_raw.csv")
    if stocks_raw_path.exists() and not force_download:
        print("Loading existing raw multi-stock dataset...")
        return pd.read_csv(stocks_raw_path)

    print("Downloading raw multi-stock dataset...")
    stocks_raw_df = download_stock_prices()
    stocks_raw_df.to_csv(stocks_raw_path, index=False)
    print(f"Saved raw multi-stock dataset to {stocks_raw_path}")
    return stocks_raw_df


def main(force_download: bool = False, run_finbert: bool = True) -> None:
    """Run the full dataset pipeline and save raw + processed artifacts."""

    data_raw_dir().mkdir(parents=True, exist_ok=True)
    data_processed_dir().mkdir(parents=True, exist_ok=True)

    news_raw_df = build_news_raw(force_download=force_download)
    news_preprocessed_df = build_news_preprocessed(news_raw_df)

    stocks_raw_df = build_stocks_raw(force_download=force_download)
    build_stocks_preprocessed(stocks_raw_df)

    sp500_raw_df = build_sp500_raw(force_download=force_download)
    sp500_preprocessed_df = build_sp500_preprocessed(sp500_raw_df)

    news_finbert_df = build_finbert_scored_with_cache(
        news_preprocessed_df,
        run_finbert=run_finbert,
    )
    news_aligned_df, news_carryover_df = build_news_aligned(
        news_finbert_df,
        sp500_preprocessed_df,
    )
    news_daily_df = build_news_daily_features(news_aligned_df, news_carryover_df)

    sp500_targets_df = build_sp500_targets(sp500_preprocessed_df)
    build_model_dataset(news_daily_df, sp500_targets_df)


if __name__ == "__main__":
    main(force_download=True, run_finbert=True)