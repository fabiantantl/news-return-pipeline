"""Block bootstrap for relative MAE improvement confidence intervals.

Implements the success-criterion bootstrap described in the project spec:
  - 1000 resamples
  - block size 5 (matches the 5-day forward-return horizon)
  - percentile method
  - random seed 0

The block bootstrap is appropriate here because consecutive trading-day
residuals are autocorrelated: each 5-day forward return overlaps with
four of its neighbours, so i.i.d. resampling would understate variance.
"""

from __future__ import annotations

import numpy as np

from .metrics import mean_absolute_error


def block_bootstrap_relative_improvement(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_model: np.ndarray,
    n_resamples: int = 1000,
    block_size: int = 5,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> dict[str, float]:
    """
    Estimate a one-sided percentile CI for relative MAE improvement.

    Ir = (MAE_baseline - MAE_model) / MAE_baseline

    Returns a dict with keys:
      - ir_point: point estimate of Ir on the full sample
      - ci_lower: lower bound of the (confidence_level) CI
      - ci_upper: upper bound of the (confidence_level) CI

    Parameters
    ----------
    y_true:
        Ground-truth target values, ordered by date.
    y_pred_baseline:
        Baseline model predictions (same order as y_true).
    y_pred_model:
        Text/improved model predictions (same order as y_true).
    n_resamples:
        Number of bootstrap resamples (default 1000).
    block_size:
        Length of each contiguous block drawn during resampling (default 5).
    confidence_level:
        Coverage probability for the interval (default 0.95).
    random_seed:
        Seed for NumPy's default random generator (default 0).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_baseline = np.asarray(y_pred_baseline, dtype=float)
    y_pred_model = np.asarray(y_pred_model, dtype=float)

    n = len(y_true)
    if not (n == len(y_pred_baseline) == len(y_pred_model)):
        raise ValueError("y_true, y_pred_baseline, and y_pred_model must have equal length.")
    if n == 0:
        raise ValueError("Input arrays must not be empty.")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1.")

    # Point estimate
    mae_baseline = mean_absolute_error(y_true, y_pred_baseline)
    mae_model = mean_absolute_error(y_true, y_pred_model)

    if mae_baseline == 0.0:
        raise ValueError(
            "Baseline MAE is exactly 0. Relative improvement is undefined."
        )

    ir_point = (mae_baseline - mae_model) / mae_baseline

    # Block bootstrap resampling
    rng = np.random.default_rng(random_seed)
    ir_boot = np.empty(n_resamples)

    # Build list of valid block starting indices
    max_start = n - block_size
    if max_start < 0:
        # Edge case: sample is shorter than one block; fall back to i.i.d.
        for b in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            ir_boot[b] = _compute_ir(y_true[idx], y_pred_baseline[idx], y_pred_model[idx])
    else:
        for b in range(n_resamples):
            indices = _draw_block_indices(rng, n, block_size, max_start)
            ir_boot[b] = _compute_ir(
                y_true[indices], y_pred_baseline[indices], y_pred_model[indices]
            )

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(ir_boot, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(ir_boot, 100 * (1 - alpha / 2)))

    return {
        "ir_point": ir_point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_resamples": n_resamples,
        "block_size": block_size,
        "confidence_level": confidence_level,
    }


def _draw_block_indices(
    rng: np.random.Generator,
    n: int,
    block_size: int,
    max_start: int,
) -> np.ndarray:
    """Draw block-bootstrap indices to approximately cover n observations."""
    indices = []
    while len(indices) < n:
        start = int(rng.integers(0, max_start + 1))
        indices.extend(range(start, min(start + block_size, n)))
    return np.array(indices[:n], dtype=int)


def _compute_ir(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_model: np.ndarray,
) -> float:
    """Compute Ir = (MAE_baseline - MAE_model) / MAE_baseline for one bootstrap sample."""
    mae_b = mean_absolute_error(y_true, y_pred_baseline)
    mae_m = mean_absolute_error(y_true, y_pred_model)
    if mae_b == 0.0:
        return 0.0
    return (mae_b - mae_m) / mae_b