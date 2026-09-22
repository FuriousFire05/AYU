import pandas as pd

from ayu.data import add_training_rul, split_engine_ids


def test_training_rul_counts_down_to_zero() -> None:
    frame = pd.DataFrame({"unit_id": [1, 1, 1, 1], "cycle": [1, 2, 3, 4]})

    result = add_training_rul(frame)

    assert result["rul"].tolist() == [3, 2, 1, 0]


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
