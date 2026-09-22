from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


class RegressionMetrics(TypedDict):
    """Metrics reported for each RUL model."""

    mae: float
    rmse: float
    r2: float
    nasa_score: float


def clip_rul_predictions(predicted_rul: ArrayLike) -> NDArray[np.float64]:
    """Convert predictions to a one-dimensional array and clip them at zero."""

    predictions = np.asarray(predicted_rul, dtype=np.float64).reshape(-1)
    return np.clip(predictions, a_min=0.0, a_max=None)


def nasa_prognostic_score(true_rul: ArrayLike, predicted_rul: ArrayLike) -> float:
    """Calculate NASA's asymmetric prognostic score.

    Positive errors are unsafe RUL overestimates and receive the steeper penalty.
    """

    actual = np.asarray(true_rul, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_rul, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("true_rul and predicted_rul must have the same shape.")

    error = predicted - actual
    penalties = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )
    return float(penalties.sum())


def evaluate_regression(
    true_rul: ArrayLike,
    predicted_rul: ArrayLike,
) -> RegressionMetrics:
    """Evaluate non-negative RUL predictions with all CIA1 metrics."""

    actual = np.asarray(true_rul, dtype=np.float64).reshape(-1)
    predicted = clip_rul_predictions(predicted_rul)
    if actual.shape != predicted.shape:
        raise ValueError("true_rul and predicted_rul must have the same shape.")

    return RegressionMetrics(
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(root_mean_squared_error(actual, predicted)),
        r2=float(r2_score(actual, predicted)),
        nasa_score=nasa_prognostic_score(actual, predicted),
    )
