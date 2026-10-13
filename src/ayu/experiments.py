from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ayu.data import (
    SENSOR_COLS,
    SUBSET_SPECS,
    add_training_rul,
    load_cmapss_frames,
    load_cmapss_test_rul,
    normalize_subset,
    split_engine_ids,
)
from ayu.eda import (
    save_lifetime_distribution,
    save_rul_distribution,
    save_sensor_rul_correlations,
)
from ayu.features import (
    ROLLING_WINDOWS,
    add_causal_rolling_features,
    determine_informative_sensors,
    get_model_feature_columns,
)
from ayu.metrics import RegressionMetrics, clip_rul_predictions, evaluate_regression
from ayu.models import (
    fit_xgboost_regressor,
    make_xgboost_regressor,
    predict_mean_rul,
)

MODEL_WITH_CYCLE = "xgboost_with_cycle"
MODEL_WITHOUT_CYCLE = "xgboost_without_cycle"
NAIVE_MODEL = "naive_mean"


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible configuration for one classical subset experiment."""

    subset: str
    random_seed: int = 42
    validation_size: float = 0.20
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS
    variance_threshold: float = 1e-10
    max_estimators: int = 3_000
    early_stopping_rounds: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "subset", normalize_subset(self.subset))
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between zero and one.")
        if not self.rolling_windows or any(w <= 0 for w in self.rolling_windows):
            raise ValueError("rolling_windows must contain positive integers.")


@dataclass(frozen=True)
class SubsetExperimentResult:
    subset: str
    output_dir: Path
    validation_metrics: dict[str, RegressionMetrics]
    selected_model: str
    selected_rounds: int
    endpoint_metrics: dict[str, RegressionMetrics] | None


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )


def _metrics_payload(
    metrics: dict[str, RegressionMetrics],
    *,
    evaluation_unit: str,
    observations: int,
) -> dict[str, Any]:
    return {
        "evaluation_unit": evaluation_unit,
        "observations": observations,
        "metrics": metrics,
    }


def _save_validation_plot(
    metrics: dict[str, RegressionMetrics], output_dir: Path, subset: str
) -> None:
    labels = list(metrics)
    values = [metrics[name]["rmse"] for name in labels]
    display = [name.replace("_", "\n") for name in labels]
    plt.figure(figsize=(8, 5))
    plt.bar(display, values)
    plt.ylabel("Cycle-level RMSE (cycles)")
    plt.title(f"{subset} Engine-disjoint Validation")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_rmse.png", dpi=200)
    plt.close()


def _save_endpoint_plot(
    predictions: pd.DataFrame, output_dir: Path, subset: str
) -> None:
    actual = predictions["actual_rul"].to_numpy(dtype=float)
    predicted = predictions["selected_xgboost_prediction"].to_numpy(dtype=float)
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, alpha=0.75)
    plt.plot([lower, upper], [lower, upper], "--", color="black", label="Ideal")
    plt.xlabel("Actual endpoint RUL (cycles)")
    plt.ylabel("Predicted endpoint RUL (cycles)")
    plt.title(f"{subset} Official Test Endpoints")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "official_endpoint_predictions.png", dpi=200)
    plt.close()


def _split_manifest(
    train: pd.DataFrame, train_ids: list[int], validation_ids: list[int]
) -> pd.DataFrame:
    partition_by_id = {unit_id: "development_train" for unit_id in train_ids}
    partition_by_id.update({unit_id: "validation" for unit_id in validation_ids})
    manifest = (
        train.groupby("unit_id", sort=True)
        .agg(rows=("cycle", "size"), final_cycle=("cycle", "max"))
        .reset_index()
    )
    manifest["partition"] = manifest["unit_id"].map(partition_by_id)
    if manifest["partition"].isna().any():
        raise RuntimeError("Every training engine must appear in the split manifest.")
    return manifest[["unit_id", "partition", "rows", "final_cycle"]]


def xgboost_parameters(config: ExperimentConfig) -> dict[str, Any]:
    """Return the controlled hyperparameters recorded for both XGBoost variants."""

    return {
        "objective": "reg:squarederror",
        "max_estimators": config.max_estimators,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": config.random_seed,
        "tree_method": "hist",
        "early_stopping_rounds": config.early_stopping_rounds,
        "selection_metric": "validation_rmse",
    }


def run_subset_experiment(
    config: ExperimentConfig,
    *,
    data_dir: Path,
    output_root: Path,
    evaluate_test: bool = False,
    force_endpoint: bool = False,
) -> SubsetExperimentResult:
    """Run one leakage-safe subset experiment and persist its evidence artifacts."""

    subset = config.subset
    subset_dir = output_root / subset.lower()
    subset_dir.mkdir(parents=True, exist_ok=True)
    endpoint_metrics_path = subset_dir / "official_endpoint_metrics.json"
    if evaluate_test and endpoint_metrics_path.exists() and not force_endpoint:
        raise FileExistsError(
            f"Locked endpoint result already exists at {endpoint_metrics_path}. "
            "Use --force-endpoint only for an explicitly authorized rerun."
        )

    train, test = load_cmapss_frames(data_dir, subset)
    train = add_training_rul(train)
    train_ids, validation_ids = split_engine_ids(
        train,
        validation_size=config.validation_size,
        random_state=config.random_seed,
    )
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Development and validation engine IDs overlap.")

    development_train = train[train["unit_id"].isin(train_ids)].copy()
    validation = train[train["unit_id"].isin(validation_ids)].copy()
    informative_sensors = determine_informative_sensors(
        development_train,
        SENSOR_COLS,
        variance_threshold=config.variance_threshold,
    )
    removed_sensors = [name for name in SENSOR_COLS if name not in informative_sensors]
    sensor_variances = development_train[SENSOR_COLS].var()

    train_features = add_causal_rolling_features(
        development_train, informative_sensors, windows=config.rolling_windows
    )
    validation_features = add_causal_rolling_features(
        validation, informative_sensors, windows=config.rolling_windows
    )
    feature_columns = {
        MODEL_WITH_CYCLE: get_model_feature_columns(
            informative_sensors,
            include_cycle=True,
            windows=config.rolling_windows,
        ),
        MODEL_WITHOUT_CYCLE: get_model_feature_columns(
            informative_sensors,
            include_cycle=False,
            windows=config.rolling_windows,
        ),
    }
    if any("unit_id" in columns for columns in feature_columns.values()):
        raise RuntimeError("unit_id must never be a model predictor.")

    train_target = train_features["rul"].to_numpy(dtype=float)
    validation_target = validation_features["rul"].to_numpy(dtype=float)
    predictions: dict[str, Any] = {
        NAIVE_MODEL: predict_mean_rul(train_target, len(validation_target))
    }
    models = {}
    for model_name in (MODEL_WITH_CYCLE, MODEL_WITHOUT_CYCLE):
        columns = feature_columns[model_name]
        model = fit_xgboost_regressor(
            train_features[columns],
            train_target,
            validation_features[columns],
            validation_target,
            n_estimators=config.max_estimators,
            early_stopping_rounds=config.early_stopping_rounds,
            random_state=config.random_seed,
        )
        models[model_name] = model
        predictions[model_name] = clip_rul_predictions(
            model.predict(validation_features[columns])
        )

    validation_metrics = {
        name: evaluate_regression(validation_target, prediction)
        for name, prediction in predictions.items()
    }
    selected_model = min(
        (MODEL_WITH_CYCLE, MODEL_WITHOUT_CYCLE),
        key=lambda name: validation_metrics[name]["rmse"],
    )
    selected_rounds = int(models[selected_model].best_iteration) + 1

    manifest = _split_manifest(train, train_ids, validation_ids)
    manifest.to_csv(subset_dir / "engine_split.csv", index=False)
    validation_predictions = validation_features[["unit_id", "cycle", "rul"]].rename(
        columns={"rul": "actual_rul"}
    )
    for name, prediction in predictions.items():
        validation_predictions[f"{name}_prediction"] = prediction
    validation_predictions.to_csv(
        subset_dir / "validation_cycle_predictions.csv", index=False
    )

    spec = SUBSET_SPECS[subset]
    _write_json(
        subset_dir / "configuration.json",
        {
            **asdict(config),
            "target": "uncapped_linear_final_cycle_minus_current_cycle",
            "test_labels_used_for_model_selection": False,
            "subset_structure": asdict(spec),
            "software": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit-learn": version("scikit-learn"),
                "xgboost": version("xgboost"),
            },
            "xgboost": xgboost_parameters(config),
        },
    )
    _write_json(
        subset_dir / "sensor_selection.json",
        {
            "fit_partition": "development_train_engines_only",
            "variance_threshold": config.variance_threshold,
            "selected": informative_sensors,
            "removed": removed_sensors,
            "variances": {
                name: float(sensor_variances[name]) for name in SENSOR_COLS
            },
        },
    )
    _write_json(
        subset_dir / "feature_manifest.json",
        {
            "rolling_windows": list(config.rolling_windows),
            "rolling_is_causal": True,
            "rolling_resets_by_engine": True,
            "features": feature_columns,
            "feature_counts": {
                name: len(columns) for name, columns in feature_columns.items()
            },
        },
    )
    _write_json(
        subset_dir / "validation_metrics.json",
        _metrics_payload(
            validation_metrics,
            evaluation_unit="cycle",
            observations=len(validation_target),
        ),
    )
    _write_json(
        subset_dir / "model_selection.json",
        {
            "eligible_models": [MODEL_WITH_CYCLE, MODEL_WITHOUT_CYCLE],
            "criterion": "lowest_engine_disjoint_validation_rmse",
            "selected_model": selected_model,
            "selected_boosting_rounds": selected_rounds,
            "best_rounds": {
                name: int(model.best_iteration) + 1 for name, model in models.items()
            },
            "official_test_labels_consulted": False,
        },
    )
    save_lifetime_distribution(train, subset_dir, subset)
    save_rul_distribution(train, subset_dir, subset)
    save_sensor_rul_correlations(
        development_train, informative_sensors, subset_dir, subset
    )
    _save_validation_plot(validation_metrics, subset_dir, subset)

    endpoint_metrics: dict[str, RegressionMetrics] | None = None
    if evaluate_test:
        # The official labels are opened only after all model selection is complete.
        full_train_features = add_causal_rolling_features(
            train, informative_sensors, windows=config.rolling_windows
        )
        test_features = add_causal_rolling_features(
            test, informative_sensors, windows=config.rolling_windows
        )
        selected_columns = feature_columns[selected_model]
        final_model = make_xgboost_regressor(
            n_estimators=selected_rounds,
            early_stopping_rounds=None,
            random_state=config.random_seed,
        )
        final_model.fit(
            full_train_features[selected_columns],
            full_train_features["rul"].to_numpy(dtype=float),
            verbose=False,
        )
        endpoints = (
            test_features.groupby("unit_id", sort=True)
            .tail(1)
            .sort_values("unit_id")
        )
        test_rul = load_cmapss_test_rul(
            data_dir,
            subset,
            expected_engines=len(endpoints),
        )
        actual = test_rul.to_numpy(dtype=float)
        selected_prediction = clip_rul_predictions(
            final_model.predict(endpoints[selected_columns])
        )
        naive_prediction = predict_mean_rul(
            full_train_features["rul"].to_numpy(dtype=float), len(endpoints)
        )
        endpoint_metrics = {
            NAIVE_MODEL: evaluate_regression(actual, naive_prediction),
            selected_model: evaluate_regression(actual, selected_prediction),
        }
        endpoint_predictions = pd.DataFrame(
            {
                "unit_id": endpoints["unit_id"].to_numpy(dtype=int),
                "actual_rul": actual,
                "naive_mean_prediction": naive_prediction,
                "selected_xgboost_prediction": selected_prediction,
            }
        )
        endpoint_predictions["selected_xgboost_error"] = (
            endpoint_predictions["selected_xgboost_prediction"]
            - endpoint_predictions["actual_rul"]
        )
        endpoint_predictions.to_csv(
            subset_dir / "official_endpoint_predictions.csv", index=False
        )
        _write_json(
            endpoint_metrics_path,
            {
                **_metrics_payload(
                    endpoint_metrics,
                    evaluation_unit="last_observed_test_cycle_per_engine",
                    observations=len(endpoints),
                ),
                "selected_model": selected_model,
                "selected_boosting_rounds": selected_rounds,
                "selection_source": "engine_disjoint_cycle_level_validation_only",
                "locked_result": True,
            },
        )
        _save_endpoint_plot(endpoint_predictions, subset_dir, subset)

    return SubsetExperimentResult(
        subset=subset,
        output_dir=subset_dir,
        validation_metrics=validation_metrics,
        selected_model=selected_model,
        selected_rounds=selected_rounds,
        endpoint_metrics=endpoint_metrics,
    )
