from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ayu.data import (
    SENSOR_COLS,
    add_training_rul,
    describe_subset,
    load_cmapss,
    split_engine_ids,
)
from ayu.eda import (
    get_sensor_variances,
    save_lifetime_distribution,
    save_rul_distribution,
    save_sensor_rul_correlations,
    save_sensor_trajectories,
)
from ayu.features import (
    add_causal_rolling_features,
    determine_informative_sensors,
    get_model_feature_columns,
)
from ayu.metrics import (
    RegressionMetrics,
    clip_rul_predictions,
    evaluate_regression,
)
from ayu.models import (
    fit_xgboost_regressor,
    make_xgboost_regressor,
    predict_mean_rul,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/raw/CMaps"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"
PREDICTION_DIR = PROJECT_ROOT / "outputs/predictions"
SUBSET = "FD001"


def print_metric_table(metrics: dict[str, RegressionMetrics]) -> None:
    """Print a consistently ordered model comparison table."""

    comparison = pd.DataFrame.from_dict(metrics, orient="index")
    comparison.index.name = "model"
    comparison = comparison[["mae", "rmse", "r2", "nasa_score"]]
    print(comparison.to_string(float_format=lambda value: f"{value:,.4f}"))


def save_prediction_figure(predictions: pd.DataFrame) -> None:
    """Save official endpoint predictions against their true RUL values."""

    actual = predictions["actual_rul"].to_numpy(dtype=float)
    predicted = predictions["predicted_rul"].to_numpy(dtype=float)
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))

    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, alpha=0.75)
    plt.plot([lower, upper], [lower, upper], "--", color="black", label="Ideal: y = x")
    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title("FD001 Official Test Endpoint Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fd001_predicted_vs_actual.png", dpi=200)
    plt.close()


def main() -> None:
    print("AYU - Remaining Useful Life Prediction")
    print("=" * 40)
    print()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    train, test, test_rul = load_cmapss(DATA_DIR, SUBSET)
    train = add_training_rul(train)

    describe_subset(train, test, test_rul)

    train_ids, validation_ids = split_engine_ids(train)
    development_train = train[train["unit_id"].isin(train_ids)].copy()
    validation = train[train["unit_id"].isin(validation_ids)].copy()

    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Training and validation engine IDs overlap.")

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
    print("=== SENSOR ANALYSIS (TRAINING ENGINES ONLY) ===")
    variances = get_sensor_variances(development_train, SENSOR_COLS)
    informative_sensors = determine_informative_sensors(
        development_train,
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
        development_train,
        informative_sensors,
        FIGURE_DIR,
    )
    print()
    print("Five sensors most correlated with RUL:")
    print(correlations.head().to_string())

    strongest_sensor = str(correlations.index[0])
    save_lifetime_distribution(train, FIGURE_DIR)
    save_rul_distribution(train, FIGURE_DIR)
    save_sensor_trajectories(train, strongest_sensor, FIGURE_DIR)

    print()
    print("=== CAUSAL FEATURE ENGINEERING ===")
    train_feature_frame = add_causal_rolling_features(
        development_train,
        informative_sensors,
    )
    validation_feature_frame = add_causal_rolling_features(
        validation,
        informative_sensors,
    )
    with_cycle_columns = get_model_feature_columns(
        informative_sensors,
        include_cycle=True,
    )
    without_cycle_columns = get_model_feature_columns(
        informative_sensors,
        include_cycle=False,
    )
    print(f"Features with cycle:    {len(with_cycle_columns)}")
    print(f"Features without cycle: {len(without_cycle_columns)}")
    print("unit_id is excluded from both feature sets.")

    train_target = train_feature_frame["rul"].to_numpy(dtype=float)
    validation_target = validation_feature_frame["rul"].to_numpy(dtype=float)

    naive_predictions = predict_mean_rul(train_target, len(validation_target))
    with_cycle_model = fit_xgboost_regressor(
        train_feature_frame[with_cycle_columns],
        train_target,
        validation_feature_frame[with_cycle_columns],
        validation_target,
    )
    without_cycle_model = fit_xgboost_regressor(
        train_feature_frame[without_cycle_columns],
        train_target,
        validation_feature_frame[without_cycle_columns],
        validation_target,
    )
    with_cycle_predictions = clip_rul_predictions(
        with_cycle_model.predict(validation_feature_frame[with_cycle_columns])
    )
    without_cycle_predictions = clip_rul_predictions(
        without_cycle_model.predict(validation_feature_frame[without_cycle_columns])
    )

    validation_metrics: dict[str, RegressionMetrics] = {
        "naive_mean": evaluate_regression(validation_target, naive_predictions),
        "xgboost_with_cycle": evaluate_regression(
            validation_target, with_cycle_predictions
        ),
        "xgboost_without_cycle": evaluate_regression(
            validation_target, without_cycle_predictions
        ),
    }

    print()
    print("=== VALIDATION METRICS (ALL CYCLES) ===")
    print_metric_table(validation_metrics)
    print()
    print(
        f"Best boosting rounds (with cycle):    {with_cycle_model.best_iteration + 1}"
    )
    print(
        "Best boosting rounds (without cycle): "
        f"{without_cycle_model.best_iteration + 1}"
    )

    validation_predictions = validation_feature_frame[
        ["unit_id", "cycle", "rul"]
    ].rename(columns={"rul": "actual_rul"})
    validation_predictions["naive_prediction"] = naive_predictions
    validation_predictions["xgboost_with_cycle_prediction"] = with_cycle_predictions
    validation_predictions["xgboost_without_cycle_prediction"] = (
        without_cycle_predictions
    )
    validation_path = PREDICTION_DIR / "fd001_validation_predictions.csv"
    validation_predictions.to_csv(validation_path, index=False)

    with_cycle_rmse = validation_metrics["xgboost_with_cycle"]["rmse"]
    without_cycle_rmse = validation_metrics["xgboost_without_cycle"]["rmse"]
    if with_cycle_rmse <= without_cycle_rmse:
        selected_name = "XGBoost with cycle"
        selected_columns = with_cycle_columns
        selected_model = with_cycle_model
        selected_metrics = validation_metrics["xgboost_with_cycle"]
        alternative_metrics = validation_metrics["xgboost_without_cycle"]
    else:
        selected_name = "XGBoost without cycle"
        selected_columns = without_cycle_columns
        selected_model = without_cycle_model
        selected_metrics = validation_metrics["xgboost_without_cycle"]
        alternative_metrics = validation_metrics["xgboost_with_cycle"]

    print()
    print("=== MODEL SELECTION ===")
    print(f"Selected: {selected_name} (lower validation RMSE).")
    print(
        f"Selected NASA score: {selected_metrics['nasa_score']:,.4f}; "
        f"alternative NASA score: {alternative_metrics['nasa_score']:,.4f}."
    )

    # The validation set has completed its model-selection role. Refit the chosen
    # feature variant on all training engines using the selected boosting length;
    # the official test labels remain untouched until the final evaluation below.
    selected_rounds = selected_model.best_iteration + 1
    full_train_feature_frame = add_causal_rolling_features(
        train,
        informative_sensors,
    )
    test_feature_frame = add_causal_rolling_features(test, informative_sensors)
    final_model = make_xgboost_regressor(
        n_estimators=selected_rounds,
        early_stopping_rounds=None,
    )
    final_model.fit(
        full_train_feature_frame[selected_columns],
        full_train_feature_frame["rul"].to_numpy(dtype=float),
        verbose=False,
    )

    test_endpoints = (
        test_feature_frame.groupby("unit_id", sort=True).tail(1).sort_values("unit_id")
    )
    actual_test_rul = test_rul.to_numpy(dtype=float)
    if len(test_endpoints) != len(actual_test_rul):
        raise RuntimeError("Official test endpoints and RUL labels do not align.")

    predicted_test_rul = clip_rul_predictions(
        final_model.predict(test_endpoints[selected_columns])
    )
    test_metrics = evaluate_regression(actual_test_rul, predicted_test_rul)
    test_predictions = pd.DataFrame(
        {
            "unit_id": test_endpoints["unit_id"].to_numpy(dtype=int),
            "actual_rul": actual_test_rul,
            "predicted_rul": predicted_test_rul,
        }
    )
    test_predictions["error"] = (
        test_predictions["predicted_rul"] - test_predictions["actual_rul"]
    )
    test_path = PREDICTION_DIR / "fd001_test_predictions.csv"
    test_predictions.to_csv(test_path, index=False)
    save_prediction_figure(test_predictions)

    print()
    print("=== OFFICIAL FD001 TEST ENDPOINT METRICS ===")
    print_metric_table({selected_name: test_metrics})
    print()
    print("Saved outputs:")
    print(validation_path.resolve())
    print(test_path.resolve())
    print((FIGURE_DIR / "fd001_predicted_vs_actual.png").resolve())
    print()
    print("Saved EDA figures to:")
    print(FIGURE_DIR.resolve())


if __name__ == "__main__":
    main()
