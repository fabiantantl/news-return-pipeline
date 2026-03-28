"""Non-text baseline models for the news-return pipeline.

Implements two baselines:
  - MeanBaseline: predicts the training-set mean for every example.
  - Lag1LinearRegression: fits OLS on lag_1_return to predict target_return_5d.

Both follow a minimal sklearn-style interface: fit(X, y) / predict(X).
"""

from __future__ import annotations

import numpy as np


class MeanBaseline:
    """Predict the training-set mean of the target for every example."""

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanBaseline":
        """Store the training mean; X is ignored."""
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("MeanBaseline has not been fitted yet. Call fit() first.")
        return np.full(len(X), self._mean)

    @property
    def train_mean(self) -> float:
        if self._mean is None:
            raise RuntimeError("MeanBaseline has not been fitted yet.")
        return self._mean


class Lag1LinearRegression:
    """
    Ordinary least-squares linear regression on a single feature: lag_1_return.

    Equivalent to:
        target_return_5d[t] = beta_0 + beta_1 * lag_1_return[t] + epsilon

    Uses the closed-form OLS solution to stay dependency-free (no sklearn required).
    """

    def __init__(self) -> None:
        self._intercept: float | None = None
        self._slope: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Lag1LinearRegression":
        """
        Fit OLS on X (shape [n, 1] or [n,]) and y (shape [n,]).
        Uses closed-form solution: [beta_0, beta_1] = (A^T A)^{-1} A^T y
        where A = [1, X] (design matrix with intercept column).
        """
        x = np.asarray(X, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        if len(x) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got {len(x)} and {len(y)}."
            )

        A = np.column_stack([np.ones(len(x)), x])
        # Closed-form OLS: (A^T A)^{-1} A^T y
        try:
            coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"OLS fit failed: {e}") from e

        self._intercept = float(coeffs[0])
        self._slope = float(coeffs[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._intercept is None or self._slope is None:
            raise RuntimeError(
                "Lag1LinearRegression has not been fitted yet. Call fit() first."
            )
        x = np.asarray(X, dtype=float).ravel()
        return self._intercept + self._slope * x

    @property
    def intercept(self) -> float:
        if self._intercept is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._intercept

    @property
    def slope(self) -> float:
        if self._slope is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._slope