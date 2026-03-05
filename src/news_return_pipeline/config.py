"""Configuration objects for the pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Minimal configuration for preprocessing and temporal splitting."""

    dataset_name: str = "custom"
    raw_filename: str = "kaggle_findkg_news_clean.csv"
    processed_filename: str = "daily_agg.csv"
    k_forward: int = 5
    train_years: tuple[int, int] = (2010, 2017)
    val_years: tuple[int, int] = (2018, 2019)
    test_years: tuple[int, int] = (2020, 2021)
    random_seed: int = 0
    min_rows_per_split: int = 30
