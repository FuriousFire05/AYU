from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from xgboost import XGBRegressor


def predict_mean_rul(
    training_rul: ArrayLike,
    n_predictions: int,
) -> NDArray[np.float64]:
    """Predict the training-set mean RUL for every requested observation."""

    values = np.asarray(training_rul, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("training_rul must contain at least one target.")
    if n_predictions < 0:
        raise ValueError("n_predictions must be non-negative.")
    return np.full(n_predictions, values.mean(), dtype=np.float64)


def make_xgboost_regressor(
    *,
    n_estimators: int = 3_000,
    early_stopping_rounds: int | None = 50,
) -> XGBRegressor:
    """Create the deterministic XGBoost regressor used in CIA1."""

    parameters: dict[str, object] = {
        "objective": "reg:squarederror",
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if early_stopping_rounds is not None:
        parameters["early_stopping_rounds"] = early_stopping_rounds
    return XGBRegressor(**parameters)


def fit_xgboost_regressor(
    train_features: pd.DataFrame,
    train_rul: ArrayLike,
    validation_features: pd.DataFrame,
    validation_rul: ArrayLike,
) -> XGBRegressor:
    """Fit XGBoost with validation-based early stopping (XGBoost 3.4 API)."""

    model = make_xgboost_regressor()
    model.fit(
        train_features,
        train_rul,
        eval_set=[(validation_features, validation_rul)],
        verbose=False,
    )
    return model
