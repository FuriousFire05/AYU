from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ayu.data import SUBSET_SPECS, normalize_subset
from ayu.experiments import (
    ExperimentConfig,
    SubsetExperimentResult,
    run_subset_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _subset_name(value: str) -> str:
    try:
        return normalize_subset(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe AYU classical baselines on C-MAPSS subsets."
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        type=_subset_name,
        default=list(SUBSET_SPECS),
        help="Subsets to run (default: FD001 FD002 FD003 FD004).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/CMaps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/baselines",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Unlock one-time official endpoint evaluation after model selection.",
    )
    parser.add_argument(
        "--force-endpoint",
        action="store_true",
        help="Replace an existing locked endpoint result (requires --evaluate-test).",
    )
    args = parser.parse_args()
    if args.force_endpoint and not args.evaluate_test:
        parser.error("--force-endpoint requires --evaluate-test")
    return args


def _summary_rows(
    results: list[SubsetExperimentResult], *, endpoint: bool
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        metrics_by_model = (
            result.endpoint_metrics if endpoint else result.validation_metrics
        )
        if metrics_by_model is None:
            continue
        for model, metrics in metrics_by_model.items():
            rows.append(
                {
                    "subset": result.subset,
                    "model": model,
                    "selected": model == result.selected_model,
                    "selected_rounds": (
                        result.selected_rounds if model == result.selected_model else None
                    ),
                    **metrics,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[SubsetExperimentResult] = []
    for subset in args.subsets:
        print(f"\n=== {subset} ===", flush=True)
        result = run_subset_experiment(
            ExperimentConfig(subset=subset),
            data_dir=args.data_dir,
            output_root=args.output_dir,
            evaluate_test=args.evaluate_test,
            force_endpoint=args.force_endpoint,
        )
        results.append(result)
        selected = result.validation_metrics[result.selected_model]
        print(
            f"selected={result.selected_model}, rounds={result.selected_rounds}, "
            f"validation_rmse={selected['rmse']:.4f}",
            flush=True,
        )
        if result.endpoint_metrics is not None:
            endpoint = result.endpoint_metrics[result.selected_model]
            print(
                f"official_endpoint_rmse={endpoint['rmse']:.4f}, "
                f"nasa_score={endpoint['nasa_score']:.4f}",
                flush=True,
            )

    validation_rows = _summary_rows(results, endpoint=False)
    pd.DataFrame(validation_rows).to_csv(
        args.output_dir / "summary_validation_cycle_metrics.csv", index=False
    )
    if args.evaluate_test:
        endpoint_rows = _summary_rows(results, endpoint=True)
        pd.DataFrame(endpoint_rows).to_csv(
            args.output_dir / "summary_official_endpoint_metrics.csv", index=False
        )
    print(f"\nArtifacts saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
