from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_lifetime_distribution(
    train: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot the distribution of engine lifetimes."""

    life_lengths = train.groupby("unit_id")["cycle"].max()

    plt.figure(figsize=(8, 5))
    plt.hist(life_lengths, bins=20)

    plt.xlabel("Observed lifetime (cycles)")
    plt.ylabel("Number of engines")
    plt.title("FD001 Training Engine Lifetime Distribution")

    plt.tight_layout()
    plt.savefig(
        output_dir / "fd001_lifetime_distribution.png",
        dpi=200,
    )
    plt.close()


def save_rul_distribution(
    train: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot the distribution of RUL targets."""

    plt.figure(figsize=(8, 5))
    plt.hist(train["rul"], bins=30)

    plt.xlabel("Remaining Useful Life (cycles)")
    plt.ylabel("Number of observations")
    plt.title("FD001 Remaining Useful Life Distribution")

    plt.tight_layout()
    plt.savefig(
        output_dir / "fd001_rul_distribution.png",
        dpi=200,
    )
    plt.close()


def get_sensor_variances(
    train: pd.DataFrame,
    sensor_cols: list[str],
) -> pd.Series:
    """Return sensor variances in descending order."""

    variances = train[sensor_cols].var()

    return variances.sort_values(ascending=False)


def get_informative_sensors(
    train: pd.DataFrame,
    sensor_cols: list[str],
    variance_threshold: float = 1e-10,
) -> list[str]:
    """Remove effectively constant sensor channels."""

    variances = train[sensor_cols].var()

    return variances[variances > variance_threshold].index.tolist()


def save_sensor_trajectories(
    train: pd.DataFrame,
    sensor: str,
    output_dir: Path,
    engine_ids: tuple[int, ...] = (1, 2, 3),
) -> None:
    """Plot one sensor across several engine lifetimes."""

    plt.figure(figsize=(9, 5))

    for unit_id in engine_ids:
        engine = train[train["unit_id"] == unit_id]

        plt.plot(
            engine["cycle"],
            engine[sensor],
            label=f"Engine {unit_id}",
        )

    plt.xlabel("Cycle")
    plt.ylabel(sensor)
    plt.title(f"FD001 Example Degradation Trajectories: {sensor}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        output_dir / f"fd001_{sensor}_trajectories.png",
        dpi=200,
    )
    plt.close()


def save_sensor_rul_correlations(
    train: pd.DataFrame,
    sensor_cols: list[str],
    output_dir: Path,
) -> pd.Series:
    """Plot absolute Pearson correlation between sensors and RUL."""

    correlations = (
        train[sensor_cols + ["rul"]]
        .corr(numeric_only=True)["rul"]
        .drop("rul")
        .abs()
        .sort_values(ascending=False)
    )

    labels = correlations.index.astype(str).tolist()
    values = correlations.to_numpy(dtype=float)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)

    plt.xlabel("Sensor")
    plt.ylabel("Absolute Pearson correlation with RUL")
    plt.title("FD001 Sensor Association with Remaining Useful Life")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(
        output_dir / "fd001_sensor_rul_correlations.png",
        dpi=200,
    )
    plt.close()

    return correlations
