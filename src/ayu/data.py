from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


def load_cmapss(
    root: Path,
    subset: str = "FD001",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load one NASA C-MAPSS subset."""

    train_path = root / f"train_{subset}.txt"
    test_path = root / f"test_{subset}.txt"
    rul_path = root / f"RUL_{subset}.txt"

    for path in (train_path, test_path, rul_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required C-MAPSS file: {path}"
            )

    train = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    test = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    test_rul = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["rul_at_last_observation"],
    )["rul_at_last_observation"]

    train = (
        train
        .sort_values(["unit_id", "cycle"])
        .reset_index(drop=True)
    )

    test = (
        test
        .sort_values(["unit_id", "cycle"])
        .reset_index(drop=True)
    )

    if train.isna().any().any():
        raise ValueError("Training data contains unexpected missing values.")

    if test.isna().any().any():
        raise ValueError("Test data contains unexpected missing values.")

    if test["unit_id"].nunique() != len(test_rul):
        raise ValueError(
            "Number of test engines does not match the number of RUL labels."
        )

    return train, test, test_rul


def add_training_rul(
    df: pd.DataFrame,
    cap: int | None = None,
) -> pd.DataFrame:
    """
    Add Remaining Useful Life labels to training trajectories.

    Because training engines run until failure:

        RUL = final cycle of engine - current cycle
    """

    result = df.copy()

    final_cycle = (
        result
        .groupby("unit_id")["cycle"]
        .transform("max")
    )

    result["rul_raw"] = final_cycle - result["cycle"]

    if cap is None:
        result["rul"] = result["rul_raw"]
    else:
        result["rul"] = result["rul_raw"].clip(upper=cap)

    return result


def split_engine_ids(
    df: pd.DataFrame,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Split by complete engines rather than individual rows.

    This prevents cycles from the same engine appearing in both
    training and validation sets.
    """

    engine_ids = df["unit_id"].unique()

    train_ids, validation_ids = train_test_split(
        engine_ids,
        test_size=validation_size,
        random_state=random_state,
    )

    return train_ids.tolist(), validation_ids.tolist()


def describe_subset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_rul: pd.Series,
) -> None:
    """Print useful sanity checks for a loaded C-MAPSS subset."""

    print("=== C-MAPSS SUMMARY ===")
    print()

    print(f"Training rows:     {len(train):,}")
    print(f"Training engines:  {train['unit_id'].nunique():,}")

    print(f"Test rows:         {len(test):,}")
    print(f"Test engines:      {test['unit_id'].nunique():,}")

    print(f"Test RUL labels:   {len(test_rul):,}")

    print()
    print(f"Columns:           {len(train.columns)}")

    print()
    print("Training lifetime statistics:")
    print(
        train.groupby("unit_id")["cycle"]
        .max()
        .describe()
        .round(2)
    )
