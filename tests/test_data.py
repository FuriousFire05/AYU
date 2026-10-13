from pathlib import Path

import pandas as pd
import pytest

from ayu.data import (
    SUBSET_SPECS,
    add_training_rul,
    load_cmapss_frames,
    normalize_subset,
    split_engine_ids,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data/raw/CMaps"


def test_training_rul_counts_down_to_zero() -> None:
    frame = pd.DataFrame({"unit_id": [1, 1, 1, 1], "cycle": [1, 2, 3, 4]})

    result = add_training_rul(frame)

    assert result["rul"].tolist() == [3, 2, 1, 0]
    assert result["rul_raw"].equals(result["rul"])


def test_training_rul_is_uncapped_for_long_engine() -> None:
    frame = pd.DataFrame({"unit_id": [1] * 201, "cycle": range(1, 202)})

    result = add_training_rul(frame)

    assert result.loc[0, "rul"] == 200
    assert result.loc[200, "rul"] == 0


def test_engine_split_has_no_engine_overlap() -> None:
    frame = pd.DataFrame(
        {
            "unit_id": [unit_id for unit_id in range(1, 11) for _ in range(2)],
            "cycle": [1, 2] * 10,
        }
    )

    train_ids, validation_ids = split_engine_ids(frame)

    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids) | set(validation_ids) == set(range(1, 11))


def test_engine_split_is_repeatable() -> None:
    frame = pd.DataFrame({"unit_id": range(1, 21), "cycle": [1] * 20})

    first = split_engine_ids(frame, random_state=42)
    second = split_engine_ids(frame, random_state=42)

    assert first == second


def test_subset_names_are_canonical_and_validated() -> None:
    assert normalize_subset(" fd004 ") == "FD004"
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_subset("FD005")


@pytest.mark.parametrize("subset", list(SUBSET_SPECS))
def test_official_subset_dimensions(subset: str) -> None:
    if not DATA_DIR.exists():
        pytest.skip("NASA C-MAPSS data is not available in this checkout.")
    train, test = load_cmapss_frames(DATA_DIR, subset)
    spec = SUBSET_SPECS[subset]

    assert (len(train), train["unit_id"].nunique()) == (
        spec.training_rows,
        spec.training_engines,
    )
    assert (len(test), test["unit_id"].nunique()) == (
        spec.test_rows,
        spec.test_engines,
    )
