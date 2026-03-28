"""Train and evaluate non-text baselines on the index-level daily dataset.

Pipeline:
  1. Load data/processed/model_dataset.csv (one row per S&P 500 trading day).
  2. Compute lag_1_return from close prices using only past data.
  3. Drop rows with missing lag_1_return or missing target_return_5d.
  4. Apply temporal splits: train 2020-2022 / val 2023-2024 / test 2025.
  5. Train MeanBaseline and Lag1LinearRegression.
  6. Evaluate both on val and test using MAE, Pearson, R².
  7. Select best non-text baseline by validation MAE.
  8. Save metrics JSON and per-date prediction CSVs.

Run from repo root:
    python -m news_return_pipeline.scripts.train_baseline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from news_return_pipeline.models.baselines import Lag1LinearRegression, MeanBaseline
from news_return_pipeline.evaluation.metrics import compute_metrics
from news_return_pipeline.paths import data_processed_dir, get_processed_path


# ── Temporal split configuration ──────────────────────────────────────────────

TRAIN_YEARS = (2020, 2022)
VAL_YEARS = (2023, 2024)
TEST_YEARS = (2025, 2025)

TARGET_COL = "target_return_5d"
PRICE_COL = "close"
DATE_COL = "date"
LAG_COL = "lag_1_return"


# ── Data loading and feature engineering ──────────────────────────────────────


def load_model_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the model-ready daily dataset from disk."""

    target_path = path or get_processed_path("model_dataset.csv")
    if not target_path.exists():
        raise FileNotFoundError(
            f"Model dataset not found at {target_path}. "
            "Run `python -m news_return_pipeline.scripts.build_dataset` first."
        )

    df = pd.read_csv(target_path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    print(f"Loaded model dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")

    return df


def add_lag_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag_1_return = (close[t] / close[t-1]) - 1.

    Uses only the row's own close and the immediately preceding row's close,
    preserving strict temporal ordering — no future data is touched.
    """
    df = df.copy()
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df[LAG_COL] = df[PRICE_COL].pct_change(1)

    n_missing = int(df[LAG_COL].isna().sum())
    print(f"lag_1_return computed. Missing values (first row + any gaps): {n_missing}")

    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where lag_1_return or target_return_5d is missing."""

    before = len(df)
    df = df.dropna(subset=[LAG_COL, TARGET_COL]).reset_index(drop=True)
    dropped = before - len(df)
    print(f"Dropped {dropped} incomplete rows. Remaining: {len(df)}")

    return df


# ── Temporal splitting ─────────────────────────────────────────────────────────


def _year_range_filter(df: pd.DataFrame, year_range: tuple[int, int]) -> pd.DataFrame:
    """Return rows whose year falls within [start, end] inclusive."""
    start, end = year_range
    mask = (df["year"] >= start) & (df["year"] <= end)
    return df.loc[mask].copy().reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply year-based temporal splits with validation checks.

    Raises ValueError if any split is empty or temporal ordering is violated.
    """
    df = df.copy()
    if "year" not in df.columns:
        df["year"] = df[DATE_COL].dt.year

    train_df = _year_range_filter(df, TRAIN_YEARS)
    val_df = _year_range_filter(df, VAL_YEARS)
    test_df = _year_range_filter(df, TEST_YEARS)

    for name, split in (("train", train_df), ("val", val_df), ("test", test_df)):
        if split.empty:
            raise ValueError(
                f"{name} split is empty for year range {TRAIN_YEARS if name == 'train' else VAL_YEARS if name == 'val' else TEST_YEARS}. "
                "Check that the dataset covers the expected years."
            )

    # Temporal ordering sanity checks
    if not (train_df[DATE_COL].max() < val_df[DATE_COL].min()):
        raise ValueError("Temporal ordering violated: max(train.date) >= min(val.date).")
    if not (val_df[DATE_COL].max() < test_df[DATE_COL].min()):
        raise ValueError("Temporal ordering violated: max(val.date) >= min(test.date).")

    print(
        f"Train: {len(train_df)} rows  ({train_df[DATE_COL].min().date()} → {train_df[DATE_COL].max().date()})"
    )
    print(
        f"Val:   {len(val_df)} rows  ({val_df[DATE_COL].min().date()} → {val_df[DATE_COL].max().date()})"
    )
    print(
        f"Test:  {len(test_df)} rows  ({test_df[DATE_COL].min().date()} → {test_df[DATE_COL].max().date()})"
    )

    return train_df, val_df, test_df


# ── Training ───────────────────────────────────────────────────────────────────


def train_baselines(
    train_df: pd.DataFrame,
) -> tuple[MeanBaseline, Lag1LinearRegression]:
    """Fit MeanBaseline and Lag1LinearRegression on training data."""

    X_train = train_df[LAG_COL].values
    y_train = train_df[TARGET_COL].values

    mean_model = MeanBaseline().fit(X_train, y_train)
    lag1_model = Lag1LinearRegression().fit(X_train, y_train)

    print(f"MeanBaseline train_mean: {mean_model.train_mean:.6f}")
    print(
        f"Lag1LinearRegression intercept: {lag1_model.intercept:.6f}, "
        f"slope: {lag1_model.slope:.6f}"
    )

    return mean_model, lag1_model


# ── Evaluation ─────────────────────────────────────────────────────────────────


def evaluate_split(
    split_df: pd.DataFrame,
    mean_model: MeanBaseline,
    lag1_model: Lag1LinearRegression,
    split_name: str,
) -> dict[str, dict[str, float]]:
    """
    Evaluate both baselines on a single split.

    Returns a dict mapping baseline name → metrics dict (mae, pearson, r2).
    """
    X = split_df[LAG_COL].values
    y_true = split_df[TARGET_COL].values

    pred_mean = mean_model.predict(X)
    pred_lag1 = lag1_model.predict(X)

    metrics_mean = compute_metrics(y_true, pred_mean)
    metrics_lag1 = compute_metrics(y_true, pred_lag1)

    print(f"\n── {split_name.upper()} ──────────────────────────────")
    print(
        f"  MeanBaseline  │ MAE {metrics_mean['mae']:.6f} │ "
        f"Pearson {metrics_mean['pearson']:+.4f} │ R² {metrics_mean['r2']:+.4f}"
    )
    print(
        f"  Lag1LinReg    │ MAE {metrics_lag1['mae']:.6f} │ "
        f"Pearson {metrics_lag1['pearson']:+.4f} │ R² {metrics_lag1['r2']:+.4f}"
    )

    return {
        "mean_baseline": metrics_mean,
        "lag1_linear": metrics_lag1,
    }


def select_best_baseline(
    val_metrics: dict[str, dict[str, float]],
) -> str:
    """
    Select the best non-text baseline by validation MAE (lower is better).

    Returns the key of the winning baseline.
    """
    best_name = min(val_metrics, key=lambda k: val_metrics[k]["mae"])
    print(f"\nBest baseline (by val MAE): {best_name}  (MAE = {val_metrics[best_name]['mae']:.6f})")
    return best_name


# ── Persistence ────────────────────────────────────────────────────────────────


def save_predictions(
    split_df: pd.DataFrame,
    mean_model: MeanBaseline,
    lag1_model: Lag1LinearRegression,
    split_name: str,
    output_dir: Path,
) -> None:
    """Save per-date predictions for both baselines as a CSV."""

    X = split_df[LAG_COL].values

    preds_df = pd.DataFrame(
        {
            DATE_COL: split_df[DATE_COL].values,
            TARGET_COL: split_df[TARGET_COL].values,
            LAG_COL: X,
            "pred_mean_baseline": mean_model.predict(X),
            "pred_lag1_linear": lag1_model.predict(X),
        }
    )

    out_path = output_dir / f"baseline_predictions_{split_name}.csv"
    preds_df.to_csv(out_path, index=False)
    print(f"Saved {split_name} predictions to {out_path}")


def save_metrics(
    all_metrics: dict,
    best_baseline: str,
    output_dir: Path,
) -> None:
    """Save combined metrics dict (val + test, both baselines, best selection) as JSON."""

    payload = {
        "best_baseline": best_baseline,
        "splits": all_metrics,
        "config": {
            "train_years": list(TRAIN_YEARS),
            "val_years": list(VAL_YEARS),
            "test_years": list(TEST_YEARS),
            "target_column": TARGET_COL,
            "baseline_feature": LAG_COL,
        },
    }

    out_path = output_dir / "baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics to {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = data_processed_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_model_dataset()

    # 2. Feature engineering
    df = add_lag_return(df)

    # 3. Drop incomplete rows
    df = drop_incomplete_rows(df)

    # 4. Temporal split
    train_df, val_df, test_df = temporal_split(df)

    # 5. Train
    mean_model, lag1_model = train_baselines(train_df)

    # 6. Evaluate
    val_metrics = evaluate_split(val_df, mean_model, lag1_model, "validation")
    test_metrics = evaluate_split(test_df, mean_model, lag1_model, "test")

    # 7. Best baseline
    best_baseline = select_best_baseline(val_metrics)

    # 8. Save predictions
    for split_name, split_df in (("val", val_df), ("test", test_df)):
        save_predictions(split_df, mean_model, lag1_model, split_name, output_dir)

    # 9. Save metrics
    all_metrics = {"val": val_metrics, "test": test_metrics}
    save_metrics(all_metrics, best_baseline, output_dir)

    print("\nBaseline training and evaluation complete.")


if __name__ == "__main__":
    main()