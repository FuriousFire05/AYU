from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ayu.data import OP_COLS, SENSOR_COLS

ROLLING_WINDOWS = (5, 10, 20)


def determine_informative_sensors(
    train: pd.DataFrame,
    sensor_cols: Sequence[str] = SENSOR_COLS,
    variance_threshold: float = 1e-10,
) -> list[str]:
    """Select non-constant sensors using only the supplied training frame."""

    if variance_threshold < 0:
        raise ValueError("variance_threshold must be non-negative.")

    missing = sorted(set(sensor_cols) - set(train.columns))
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")

    variances = train[list(sensor_cols)].var()
    return [sensor for sensor in sensor_cols if variances[sensor] > variance_threshold]


def add_causal_rolling_features(
    frame: pd.DataFrame,
    informative_sensors: Sequence[str],
    windows: Sequence[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add per-engine trailing means and standard deviations.

    Rows are ordered by engine and cycle before calculation. Every rolling window
    contains only the current and preceding observations from the same engine.
    """

    required = {"unit_id", "cycle", *informative_sensors}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if any(window <= 0 for window in windows):
        raise ValueError("Rolling windows must be positive integers.")

    result = frame.sort_values(["unit_id", "cycle"], kind="stable").reset_index(
        drop=True
    )
    grouped = result.groupby("unit_id", sort=False)

    new_features: dict[str, pd.Series] = {}
    for sensor in informative_sensors:
        for window in windows:
            rolling = grouped[sensor].rolling(window=window, min_periods=1)
            new_features[f"{sensor}_rolling_mean_{window}"] = (
                rolling.mean().reset_index(level=0, drop=True)
            )
            new_features[f"{sensor}_rolling_std_{window}"] = (
                rolling.std().reset_index(level=0, drop=True).fillna(0.0)
            )

    if new_features:
        result = pd.concat([result, pd.DataFrame(new_features)], axis=1)

    return result


def get_model_feature_columns(
    informative_sensors: Sequence[str],
    *,
    include_cycle: bool,
    windows: Sequence[int] = ROLLING_WINDOWS,
) -> list[str]:
    """Return the controlled CIA1 model feature list."""

    columns = list(OP_COLS)
    if include_cycle:
        columns.insert(0, "cycle")
    columns.extend(informative_sensors)

    for sensor in informative_sensors:
        for window in windows:
            columns.extend(
                [
                    f"{sensor}_rolling_mean_{window}",
                    f"{sensor}_rolling_std_{window}",
                ]
            )

    return columns
