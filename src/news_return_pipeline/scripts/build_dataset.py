"""Build raw and processed datasets for the news-return pipeline."""

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


def _load_or_download_csv(
    path,
    download_fn,
    *,
    force_download: bool,
    dataset_label: str,
) -> pd.DataFrame:
    if path.exists() and not force_download:
        print(f"Loading existing raw {dataset_label} dataset...")
        return pd.read_csv(path)

    print(f"Downloading raw {dataset_label} dataset...")
    df = download_fn()
    df.to_csv(path, index=False)
    print(f"Saved raw {dataset_label} dataset to {path}")
    return df


def build_news_raw(force_download: bool = False) -> pd.DataFrame:
    news_raw_path = get_raw_path("news_raw.csv")
    return _load_or_download_csv(
        news_raw_path,
        download_news_headlines,
        force_download=force_download,
        dataset_label="news",
    )


def build_news_preprocessed(news_raw_df: pd.DataFrame) -> pd.DataFrame:
    news_preprocessed_df = preprocess_news_dataframe(news_raw_df)
    output_path = get_processed_path("news_preprocessed.csv")
    news_preprocessed_df.to_csv(output_path, index=False)
    print(f"Saved preprocessed news dataset to {output_path}")
    return news_preprocessed_df


def build_finbert_scored_with_cache(
    news_preprocessed_df: pd.DataFrame,
    *,
    run_finbert: bool,
) -> pd.DataFrame:
    cache_path = get_processed_path("news_finbert_scored.csv")

    cache_columns = list(news_preprocessed_df.columns) + ["sentiment_label", "sentiment_score"]
    cache_key = ["date", "title"]

    news_preprocessed_df = news_preprocessed_df.copy()
    news_preprocessed_df["date"] = pd.to_datetime(
        news_preprocessed_df["date"], errors="coerce"
    ).dt.normalize()

    if cache_path.exists():
        print(f"Loading FinBERT cache from {cache_path}")
        news_finbert_df = pd.read_csv(cache_path)
        news_finbert_df["date"] = pd.to_datetime(news_finbert_df["date"], errors="coerce").dt.normalize()
        missing_columns = [c for c in cache_columns if c not in news_finbert_df.columns]
        if missing_columns:
            raise ValueError(
                "FinBERT cache is missing required columns: "
                f"{missing_columns}. Delete {cache_path} and rerun."
            )
        news_finbert_df = news_finbert_df.loc[:, cache_columns]
    else:
        print("No existing FinBERT cache found. Starting a new cache.")
        news_finbert_df = pd.DataFrame(columns=cache_columns)

    missing_mask = ~news_preprocessed_df.set_index(cache_key).index.isin(
        news_finbert_df.set_index(cache_key).index
    )
    news_to_score_df = news_preprocessed_df.loc[missing_mask].copy()

    if len(news_to_score_df) > 0:
        if not run_finbert:
            raise FileNotFoundError(
                "FinBERT cache does not fully cover news_preprocessed.csv, "
                "and run_finbert=False. Rerun with run_finbert=True."
            )

        print(f"Running FinBERT for {len(news_to_score_df)} missing headline row(s)...")
        new_scored_df = compute_finbert_sentiment(news_to_score_df, text_column="title")

        news_finbert_df = pd.concat([news_finbert_df, new_scored_df], ignore_index=True)
        news_finbert_df = news_finbert_df.drop_duplicates(subset=cache_key, keep="last")
        news_finbert_df = news_finbert_df.sort_values(cache_key).reset_index(drop=True)
        news_finbert_df.to_csv(cache_path, index=False)
        print(f"Updated FinBERT cache at {cache_path}")
    else:
        print("FinBERT cache already covers all preprocessed headlines.")

    current_finbert_df = (
        news_finbert_df.merge(news_preprocessed_df[cache_key], on=cache_key, how="inner")
        .drop_duplicates(subset=cache_key)
        .sort_values(cache_key)
        .reset_index(drop=True)
    )

    return current_finbert_df


def build_news_aligned(news_finbert_df: pd.DataFrame, sp500_preprocessed_df: pd.DataFrame) -> pd.DataFrame:
    news_aligned_df = align_news_to_trading_calendar(
        news_finbert_df,
        sp500_preprocessed_df["date"],
    )
    aligned_path = get_processed_path("news_aligned.csv")
    news_aligned_df.to_csv(aligned_path, index=False)
    print(f"Saved aligned headline dataset to {aligned_path}")

    carryover_df = build_carryover_features(news_aligned_df)
    carryover_path = get_processed_path("news_carryover_features.csv")
    carryover_df.to_csv(carryover_path, index=False)
    print(f"Saved carryover feature dataset to {carryover_path}")

    return news_aligned_df


def build_news_daily_features(news_aligned_df: pd.DataFrame) -> pd.DataFrame:
    news_daily_df = build_daily_news_features(news_aligned_df)

    carryover_path = get_processed_path("news_carryover_features.csv")
    carryover_df = pd.read_csv(carryover_path)
    carryover_df["date"] = pd.to_datetime(carryover_df["date"], errors="coerce").dt.normalize()

    news_daily_df = (
        news_daily_df.merge(carryover_df, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    news_daily_df["n_carryover_headlines"] = news_daily_df["n_carryover_headlines"].fillna(0).astype(int)
    news_daily_df["has_carryover_news"] = news_daily_df["has_carryover_news"].fillna(0).astype(int)

    daily_path = get_processed_path("news_daily_features.csv")
    news_daily_df.to_csv(daily_path, index=False)
    print(f"Saved daily news feature dataset to {daily_path}")

    return news_daily_df


def build_sp500_raw(force_download: bool = False) -> pd.DataFrame:
    sp500_raw_path = get_raw_path("sp500_raw.csv")
    return _load_or_download_csv(
        sp500_raw_path,
        download_sp500_index,
        force_download=force_download,
        dataset_label="S&P 500",
    )


def build_sp500_preprocessed(sp500_raw_df: pd.DataFrame) -> pd.DataFrame:
    sp500_preprocessed_df = preprocess_sp500(sp500_raw_df)
    output_path = get_processed_path("sp500_preprocessed.csv")
    sp500_preprocessed_df.to_csv(output_path, index=False)
    print(f"Saved preprocessed S&P 500 dataset to {output_path}")
    return sp500_preprocessed_df


def build_stocks_raw(force_download: bool = False) -> pd.DataFrame:
    stocks_raw_path = get_raw_path("stocks_raw.csv")
    return _load_or_download_csv(
        stocks_raw_path,
        download_stock_prices,
        force_download=force_download,
        dataset_label="multi-stock",
    )


def build_stocks_preprocessed(stocks_raw_df: pd.DataFrame) -> pd.DataFrame:
    stocks_preprocessed_df = preprocess_stocks_dataframe(stocks_raw_df)
    output_path = get_processed_path("stocks_preprocessed.csv")
    stocks_preprocessed_df.to_csv(output_path, index=False)
    print(f"Saved preprocessed multi-stock dataset to {output_path}")
    return stocks_preprocessed_df


def build_sp500_targets(sp500_preprocessed_df: pd.DataFrame) -> pd.DataFrame:
    sp500_targets_df = add_forward_return_target(sp500_preprocessed_df, horizon=5)
    output_path = get_processed_path("sp500_with_targets.csv")
    sp500_targets_df.to_csv(output_path, index=False)
    print(f"Saved S&P 500 targets dataset to {output_path}")
    return sp500_targets_df


def build_model_dataset(news_daily_df: pd.DataFrame, sp500_targets_df: pd.DataFrame) -> pd.DataFrame:
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

    output_path = get_processed_path("model_dataset.csv")
    model_df.to_csv(output_path, index=False)
    print(f"Saved final model dataset to {output_path}")
    return model_df


def main(force_download: bool = False, run_finbert: bool = True) -> None:
    """Run the full dataset pipeline and persist raw/processed stages."""

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
    news_aligned_df = build_news_aligned(news_finbert_df, sp500_preprocessed_df)
    news_daily_df = build_news_daily_features(news_aligned_df)

    sp500_targets_df = build_sp500_targets(sp500_preprocessed_df)
    build_model_dataset(news_daily_df, sp500_targets_df)


if __name__ == "__main__":
    main(force_download=False, run_finbert=True)
