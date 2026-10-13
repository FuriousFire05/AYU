from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


@dataclass(frozen=True)
class CmapssSubsetSpec:
    """Published structure of one NASA C-MAPSS subset."""

    name: str
    training_rows: int
    training_engines: int
    test_rows: int
    test_engines: int
    operating_conditions: int
    fault_modes: int


SUBSET_SPECS = {
    "FD001": CmapssSubsetSpec("FD001", 20_631, 100, 13_096, 100, 1, 1),
    "FD002": CmapssSubsetSpec("FD002", 53_759, 260, 33_991, 259, 6, 1),
    "FD003": CmapssSubsetSpec("FD003", 24_720, 100, 16_596, 100, 1, 2),
    "FD004": CmapssSubsetSpec("FD004", 61_249, 249, 41_214, 248, 6, 2),
}


def normalize_subset(subset: str) -> str:
    """Return a canonical subset name and reject unsupported values."""

    normalized = subset.strip().upper()
    if normalized not in SUBSET_SPECS:
        choices = ", ".join(SUBSET_SPECS)
        raise ValueError(f"Unsupported C-MAPSS subset {subset!r}; choose one of {choices}.")
    return normalized


def _read_trajectory(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return frame.sort_values(["unit_id", "cycle"], kind="stable").reset_index(
        drop=True
    )


def _validate_trajectory(frame: pd.DataFrame, *, label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label} trajectory data is empty.")
    if list(frame.columns) != COLUMN_NAMES:
        raise ValueError(f"{label} trajectory data does not match the C-MAPSS schema.")
    if frame.isna().any().any():
        raise ValueError(f"{label} trajectory data contains unexpected missing values.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} trajectory data contains non-finite values.")
    if (frame[["unit_id", "cycle"]] <= 0).any().any():
        raise ValueError(f"{label} unit IDs and cycles must be positive.")
    if frame.duplicated(["unit_id", "cycle"]).any():
        raise ValueError(f"{label} contains duplicate engine-cycle rows.")

    expected_cycle = frame.groupby("unit_id").cumcount() + 1
    if not frame["cycle"].reset_index(drop=True).equals(expected_cycle):
        raise ValueError(f"{label} cycles must be contiguous and start at one per engine.")


def load_cmapss_frames(
    root: Path,
    subset: str = "FD001",
    *,
    validate_expected_counts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test trajectories without opening official endpoint labels."""

    normalized = normalize_subset(subset)
    train_path = root / f"train_{normalized}.txt"
    test_path = root / f"test_{normalized}.txt"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required C-MAPSS file: {path}")

    train = _read_trajectory(train_path)
    test = _read_trajectory(test_path)
    _validate_trajectory(train, label=f"{normalized} training")
    _validate_trajectory(test, label=f"{normalized} test")

    if validate_expected_counts:
        spec = SUBSET_SPECS[normalized]
        observed = (
            len(train),
            int(train["unit_id"].nunique()),
            len(test),
            int(test["unit_id"].nunique()),
        )
        expected = (
            spec.training_rows,
            spec.training_engines,
            spec.test_rows,
            spec.test_engines,
        )
        if observed != expected:
            raise ValueError(
                f"{normalized} dimensions {observed} do not match the published "
                f"C-MAPSS dimensions {expected}."
            )

    return train, test


def load_cmapss_test_rul(
    root: Path,
    subset: str,
    *,
    expected_engines: int | None = None,
) -> pd.Series:
    """Load official endpoint labels only when final evaluation is authorized."""

    normalized = normalize_subset(subset)
    rul_path = root / f"RUL_{normalized}.txt"
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing required C-MAPSS file: {rul_path}")

    frame = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["rul_at_last_observation"],
    )
    values = cast(pd.Series, frame["rul_at_last_observation"]).reset_index(drop=True)
    if values.empty or values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{normalized} official RUL labels are empty or invalid.")
    if (values < 0).any():
        raise ValueError(f"{normalized} official RUL labels must be non-negative.")
    if expected_engines is not None and len(values) != expected_engines:
        raise ValueError(
            "Number of test engines does not match the number of supplied RUL labels."
        )
    return values


def load_cmapss(
    root: Path,
    subset: str = "FD001",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load one complete NASA C-MAPSS subset (legacy convenience API)."""

    train, test = load_cmapss_frames(root, subset)
    test_rul = load_cmapss_test_rul(
        root,
        subset,
        expected_engines=int(test["unit_id"].nunique()),
    )
    return train, test, test_rul


def add_training_rul(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add Remaining Useful Life labels.

    Training trajectories continue until failure, so:

        RUL = final engine cycle - current cycle
    """

    result = df.copy()

    final_cycle = result.groupby("unit_id")["cycle"].transform("max")

    result["rul_raw"] = final_cycle - result["cycle"]
    result["rul"] = result["rul_raw"]

    return result


def split_engine_ids(
    df: pd.DataFrame,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Split complete engines into train and validation sets.

    Cycles from one engine must never appear in both sets.
    """

    engine_ids = df["unit_id"].astype(int).drop_duplicates().tolist()

    train_ids, validation_ids = train_test_split(
        engine_ids,
        test_size=validation_size,
        random_state=random_state,
    )

    return list(train_ids), list(validation_ids)


def describe_subset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_rul: pd.Series,
) -> None:
    """Print basic sanity checks for a C-MAPSS subset."""

    print("=== C-MAPSS SUMMARY ===")
    print()

    print(f"Training rows:     {len(train):,}")
    print(f"Training engines:  {train['unit_id'].nunique():,}")
    print(f"Test rows:         {len(test):,}")
    print(f"Test engines:      {test['unit_id'].nunique():,}")
    print(f"Test RUL labels:   {len(test_rul):,}")

    print()
    print(f"Columns:           {len(train.columns)}")

    life_lengths = train.groupby("unit_id")["cycle"].max()

    print()
    print("Training lifetime statistics:")
    print(life_lengths.describe().round(2).to_string())
