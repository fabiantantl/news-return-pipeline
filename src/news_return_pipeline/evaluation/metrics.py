"""Regression evaluation metrics for the news-return pipeline.

Primary metric: MAE (mean absolute error).
Secondary metrics: Pearson correlation coefficient, R-squared.

All functions accept plain numpy arrays and return scalars or dicts.
"""

from __future__ import annotations

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shapes(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson correlation coefficient between y_true and y_pred.

    Returns 0.0 if either array has zero variance (avoids NaN in degenerate
    cases such as the mean baseline predicting a constant).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shapes(y_true, y_pred)

    std_true = np.std(y_true)
    std_pred = np.std(y_pred)

    if std_true == 0.0 or std_pred == 0.0:
        return 0.0

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot
    where SS_res = sum((y_true - y_pred)²) and SS_tot = sum((y_true - mean(y_true))²).

    Returns 0.0 if y_true has zero variance (degenerate case).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _check_shapes(y_true, y_pred)

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute all regression metrics and return as a dictionary.

    Keys: mae, pearson, r2.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "pearson": pearson_correlation(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }


def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} and {y_pred.shape}."
        )
    if len(y_true) == 0:
        raise ValueError("Input arrays must not be empty.")