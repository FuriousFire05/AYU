from pathlib import Path

from ayu.data import (
    add_training_rul,
    describe_subset,
    load_cmapss,
    split_engine_ids,
)

DATA_DIR = Path("data/raw/CMaps")
SUBSET = "FD001"


def main() -> None:
    print("AYU - Remaining Useful Life Prediction")
    print("=" * 40)
    print()

    train, test, test_rul = load_cmapss(
        DATA_DIR,
        SUBSET,
    )

    train = add_training_rul(train)

    describe_subset(
        train,
        test,
        test_rul,
    )

    train_ids, validation_ids = split_engine_ids(train)

    print()
    print("=== ENGINE SPLIT ===")
    print(f"Training engines:   {len(train_ids)}")
    print(f"Validation engines: {len(validation_ids)}")

    first_unit = int(train["unit_id"].iloc[0])

    example_engine = train.loc[
        train["unit_id"] == first_unit,
        ["unit_id", "cycle", "rul"],
    ]

    print()
    print("=== RUL SANITY CHECK ===")
    print()
    print("First five cycles:")
    print(example_engine.head().to_string(index=False))

    print()
    print("Last five cycles:")
    print(example_engine.tail().to_string(index=False))


if __name__ == "__main__":
    main()
