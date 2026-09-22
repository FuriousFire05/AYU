from pathlib import Path

from ayu.data import (
    SENSOR_COLS,
    add_training_rul,
    describe_subset,
    load_cmapss,
    split_engine_ids,
)
from ayu.eda import (
    get_informative_sensors,
    get_sensor_variances,
    save_lifetime_distribution,
    save_rul_distribution,
    save_sensor_rul_correlations,
    save_sensor_trajectories,
)

DATA_DIR = Path("data/raw/CMaps")
OUTPUT_DIR = Path("outputs/figures")
SUBSET = "FD001"


def main() -> None:
    print("AYU - Remaining Useful Life Prediction")
    print("=" * 40)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    print()
    print("=== SENSOR ANALYSIS ===")

    variances = get_sensor_variances(
        train,
        SENSOR_COLS,
    )

    informative_sensors = get_informative_sensors(
        train,
        SENSOR_COLS,
    )

    removed_sensors = sorted(set(SENSOR_COLS) - set(informative_sensors))

    print(f"Total sensors:       {len(SENSOR_COLS)}")
    print(f"Informative sensors: {len(informative_sensors)}")
    print(f"Removed sensors:     {removed_sensors}")

    print()
    print("Five highest-variance sensors:")
    print(variances.head().to_string())

    correlations = save_sensor_rul_correlations(
        train,
        informative_sensors,
        OUTPUT_DIR,
    )

    print()
    print("Five sensors most correlated with RUL:")
    print(correlations.head().to_string())

    strongest_sensor = str(correlations.index[0])

    save_lifetime_distribution(
        train,
        OUTPUT_DIR,
    )

    save_rul_distribution(
        train,
        OUTPUT_DIR,
    )

    save_sensor_trajectories(
        train,
        strongest_sensor,
        OUTPUT_DIR,
    )

    print()
    print("Saved EDA figures to:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
