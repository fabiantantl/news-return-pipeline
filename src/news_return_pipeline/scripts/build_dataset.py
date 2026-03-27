# Orchestrate:
# 1) download/load raw data from Kaggle
# 2) preprocess datasets
# 3) save output to data/processed/...

import pandas as pd

from news_return_pipeline.datasets.kaggle_news import download_news_headlines
from news_return_pipeline.datasets.kaggle_stocks import download_stock_prices
from news_return_pipeline.datasets.kaggle_sp500 import download_sp500_index
from news_return_pipeline.paths import (
    data_raw_dir,
    data_processed_dir,
    get_raw_path,
    get_processed_path,
)
from news_return_pipeline.pipeline.aggregate_news import build_daily_news_features
from news_return_pipeline.pipeline.preprocess_news import preprocess_news_dataframe
from news_return_pipeline.pipeline.preprocess_sp500 import preprocess_sp500
from news_return_pipeline.pipeline.preprocess_stocks import preprocess_stocks_dataframe
from news_return_pipeline.pipeline.make_targets import add_forward_return_target

from news_return_pipeline.pipeline.align_news_to_trading_calendar import (
    align_news_to_trading_calendar,
    build_carryover_features,
)

from news_return_pipeline.pipeline.finbert_sentiment import compute_finbert_sentiment

def main(force_download: bool = False, run_finbert: bool = False) -> None:
    """Download raw data if needed, preprocess datasets, and save outputs."""

    data_raw_dir().mkdir(parents=True, exist_ok=True)
    data_processed_dir().mkdir(parents=True, exist_ok=True)

    news_raw_path = get_raw_path("news_dataset_raw.csv")
    news_output_path = get_processed_path("news_dataset.csv")
    news_finbert_output_path = get_processed_path("news_dataset_finbert_3m.csv")
    daily_news_output_path = get_processed_path("daily_news_features_3m.csv")

    stocks_raw_path = get_raw_path("stocks_dataset_raw.csv")
    stocks_output_path = get_processed_path("stocks_dataset.csv")

    sp500_raw_path = get_raw_path("sp500_index_raw.csv")
    sp500_output_path = get_processed_path("sp500_index.csv")

    # -------------------------
    # PROTOTYPE WINDOW CONTROL
    # -------------------------
    # For prototyping:
    # keep USE_PROTOTYPE_WINDOW = True
    # keep PROTOTYPE_MONTHS = 3
    #
    # For the full dataset later:
    # set USE_PROTOTYPE_WINDOW = False
    # -------------------------
    USE_PROTOTYPE_WINDOW = True
    PROTOTYPE_MONTHS = 3

    # -------------------------
    # News raw dataset
    # -------------------------
    if news_raw_path.exists() and not force_download:
        print("Loading existing raw news dataset...")
        news_raw_df = pd.read_csv(news_raw_path)
    else:
        print("Downloading raw news dataset...")
        news_raw_df = download_news_headlines()
        news_raw_df.to_csv(news_raw_path, index=False)

    # -------------------------
    # News processed dataset
    # -------------------------
    print("Running news preprocessing from raw dataset...")
    news_df = preprocess_news_dataframe(news_raw_df)
    news_df.to_csv(news_output_path, index=False)
    print(f"Saved processed news dataset to {news_output_path}")

    # -------------------------
    # Multi-stock raw dataset
    # -------------------------
    if stocks_raw_path.exists() and not force_download:
        print("Loading existing raw stock dataset...")
        stocks_raw_df = pd.read_csv(stocks_raw_path)
    else:
        print("Downloading raw stock dataset...")
        stocks_raw_df = download_stock_prices()
        stocks_raw_df.to_csv(stocks_raw_path, index=False)

    # -------------------------
    # Multi-stock processed dataset
    # -------------------------
    print("Running stock preprocessing from raw dataset...")
    stocks_df = preprocess_stocks_dataframe(stocks_raw_df)
    stocks_df.to_csv(stocks_output_path, index=False)
    print(f"Saved processed stock dataset to {stocks_output_path}")

    # -------------------------
    # S&P 500 raw dataset
    # -------------------------
    if sp500_raw_path.exists() and not force_download:
        print("Loading existing raw S&P 500 index dataset...")
        sp500_raw_df = pd.read_csv(sp500_raw_path)
    else:
        print("Downloading raw S&P 500 index dataset...")
        sp500_raw_df = download_sp500_index()
        sp500_raw_df.to_csv(sp500_raw_path, index=False)

    # -------------------------
    # S&P 500 processed dataset
    # -------------------------
    print("Running S&P 500 preprocessing from raw dataset...")
    sp500_df = preprocess_sp500(sp500_raw_df)

    print("About to save S&P 500 dataset to:", sp500_output_path)
    try:
        sp500_df.to_csv(sp500_output_path, index=False)
        print(f"Saved processed S&P 500 dataset to {sp500_output_path}")
    except PermissionError:
        print(f"Permission denied when saving to {sp500_output_path}")
        print("Close the file if it is open in Excel or another program, then rerun.")
        raise

    # -------------------------
    # Cached FinBERT -> trading-calendar alignment -> daily aggregated news
    # -------------------------
    if run_finbert:
        if not news_finbert_output_path.exists():
            raise FileNotFoundError(
                f"Expected FinBERT file not found at {news_finbert_output_path}. "
                "Run FinBERT once first."
            )

        print("Loading cached FinBERT dataset...")
        news_finbert_df = pd.read_csv(news_finbert_output_path)

        news_finbert_df["date"] = pd.to_datetime(
            news_finbert_df["date"], errors="coerce"
        ).dt.normalize()
        if news_finbert_df["date"].isna().any():
            raise ValueError("news_finbert_df contains non-parseable dates.")

        print("Aligning headline dates to S&P 500 trading calendar...")
        news_finbert_aligned_df = align_news_to_trading_calendar(
            news_finbert_df,
            sp500_df["date"],
        )

        print("\nAligned headline preview:")
        preview_columns = ["original_date", "date", "was_mapped", "days_shifted", "title"]
        preview_columns = [c for c in preview_columns if c in news_finbert_aligned_df.columns]
        print(news_finbert_aligned_df[preview_columns].head(10).to_string(index=False))

        print("Aggregating daily news features...")
        daily_news_df = build_daily_news_features(news_finbert_aligned_df)

        print("Building carryover features...")
        carryover_df = build_carryover_features(news_finbert_aligned_df)

        daily_news_df = (
            daily_news_df.merge(carryover_df, on="date", how="left")
            .sort_values("date")
            .reset_index(drop=True)
        )

        daily_news_df["n_carryover_headlines"] = (
            daily_news_df["n_carryover_headlines"].fillna(0).astype(int)
        )
        daily_news_df["has_carryover_news"] = (
            daily_news_df["has_carryover_news"].fillna(0).astype(int)
        )

        daily_news_df.to_csv(daily_news_output_path, index=False)
        print(f"Saved daily aggregated news output to {daily_news_output_path}")

    else:
        print("Skipping aggregation from cached FinBERT file...")

        if not daily_news_output_path.exists():
            raise FileNotFoundError(
                f"Expected daily news file not found at {daily_news_output_path}. "
                "Run once with run_finbert=True to create it."
            )

        print("Loading existing daily aggregated news dataset...")
        daily_news_df = pd.read_csv(daily_news_output_path)

    # Make sure daily news date is parsed correctly after CSV load
    daily_news_df["date"] = pd.to_datetime(
        daily_news_df["date"], errors="coerce"
    ).dt.normalize()
    if daily_news_df["date"].isna().any():
        raise ValueError("daily_news_df contains non-parseable dates.")

    # Dynamic prototype window based on news coverage
    prototype_start = daily_news_df["date"].min()
    prototype_end = prototype_start + pd.DateOffset(months=PROTOTYPE_MONTHS)

    print("\nPrototype window derived from daily_news_df:")
    print("prototype_start =", prototype_start.date())
    print("prototype_end   =", (prototype_end - pd.Timedelta(days=1)).date())

    # -------------------------
    # Create forward return target on FULL S&P 500 trading-day calendar
    # -------------------------
    sp500_with_target_df = add_forward_return_target(sp500_df, horizon=5)

    # Optional debug: verify exact future date used for target
    sp500_with_target_df["future_date_5d"] = sp500_with_target_df["date"].shift(-5)
    sp500_with_target_df["calendar_day_gap"] = (
        sp500_with_target_df["future_date_5d"] - sp500_with_target_df["date"]
    ).dt.days

    print("\nS&P 500 target debug preview:")
    print(
        sp500_with_target_df[
            ["date", "future_date_5d", "calendar_day_gap", "close", "target_return_5d"]
        ].head(10).to_string(index=False)
    )

    # -------------------------
    # Merge daily news onto S&P 500 trading-day calendar
    # Keep all trading days, even if there was no news
    # -------------------------
    model_df = (
        sp500_with_target_df.merge(
            daily_news_df,
            on="date",
            how="left",
            suffixes=("", "_news"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Rebuild year from date so every trading day has a valid year
    model_df["year"] = model_df["date"].dt.year

    # Fill missing news features for no-news trading days
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

    for col in zero_fill_columns:
        if col in model_df.columns:
            model_df[col] = model_df[col].fillna(0)

    if "text" in model_df.columns:
        model_df["text"] = model_df["text"].fillna("")

    # Drop rows whose 5-day future return is unavailable
    model_df = model_df.dropna(subset=["target_return_5d"]).reset_index(drop=True)

    # -------------------------
    # PROTOTYPE TRUNCATION BLOCK
    # Edit here if you want the full dataset later:
    # - keep this block for 3-month prototyping
    # - disable it by setting USE_PROTOTYPE_WINDOW = False above
    # -------------------------
    if USE_PROTOTYPE_WINDOW:
        model_df = model_df.loc[
            (model_df["date"] >= prototype_start) & (model_df["date"] < prototype_end)
        ].reset_index(drop=True)

        print("\nUsing prototype-truncated dataset")
        print(
            "Prototype-truncated date range:",
            model_df["date"].min().date(),
            "to",
            model_df["date"].max().date(),
        )
        print("Prototype-truncated rows:", len(model_df))
    else:
        print("\nUsing full dataset")
        print(
            "Full date range:",
            model_df["date"].min().date(),
            "to",
            model_df["date"].max().date(),
        )
        print("Full rows:", len(model_df))

    # Drop helper debug columns before saving
    helper_columns = ["future_date_5d", "calendar_day_gap"]
    model_df = model_df.drop(columns=[c for c in helper_columns if c in model_df.columns])

    # Move text to last column
    if "text" in model_df.columns:
        model_df = model_df[[col for col in model_df.columns if col != "text"] + ["text"]]

    print("\nModel dataset shape:", model_df.shape)
    print("\nModel dataset headers:")
    # print(model_df.columns.tolist())
    # print("\nFirst 5 rows of model dataset:")
    # print(model_df.head().to_string(index=False))

    if USE_PROTOTYPE_WINDOW:
        model_output_path = get_processed_path("model_dataset_3m.csv")
    else:
        model_output_path = get_processed_path("model_dataset_full.csv")

    model_df.to_csv(model_output_path, index=False)

    print(f"Saved final model dataset to {model_output_path}")

    # print("\nRows whose FINAL mapped date is still weekend:")
    # print((news_finbert_aligned_df["date"].dt.weekday >= 5).sum())

    # print("\nMapped rows not landing on Monday:")
    # mapped_not_monday = news_finbert_aligned_df.loc[
    #     news_finbert_aligned_df["was_mapped"] &
    #     (news_finbert_aligned_df["date"].dt.day_name() != "Monday"),
    #     ["original_date", "date", "days_shifted", "title"]
    # ].copy()

    # mapped_not_monday["original_weekday"] = mapped_not_monday["original_date"].dt.day_name()
    # mapped_not_monday["mapped_weekday"] = mapped_not_monday["date"].dt.day_name()

    # print(
    #     mapped_not_monday[
    #         ["original_date", "original_weekday", "date", "mapped_weekday", "days_shifted", "title"]
    #     ].head(20).to_string(index=False)
    # )

    # print("\nAggregated rows with carryover news:")
    # print(
    #     daily_news_df.loc[
    #         daily_news_df["has_carryover_news"] == 1,
    #         ["date", "n_headlines", "n_carryover_headlines", "has_carryover_news"]
    #     ].head(20).to_string(index=False)
    # )

    # print("\nAlignment summary:")
    # print("Total headline rows:", len(news_finbert_aligned_df))
    # print("Mapped rows:", int(news_finbert_aligned_df["was_mapped"].sum()))
    # print(
    #     "Mapped-to-Monday rows:",
    #     int(
    #         news_finbert_aligned_df.loc[
    #             news_finbert_aligned_df["was_mapped"] &
    #             (news_finbert_aligned_df["date"].dt.day_name() == "Monday")
    #         ].shape[0]
    #     )
    # )


if __name__ == "__main__":
    main(force_download=False, run_finbert=True)

