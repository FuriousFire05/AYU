import numpy as np
import pytest

from ayu.metrics import evaluate_regression, nasa_prognostic_score


def test_nasa_score_is_zero_for_exact_predictions() -> None:
    assert nasa_prognostic_score([20.0], [20.0]) == pytest.approx(0.0)


def test_nasa_score_penalizes_overestimation_more() -> None:
    underestimated = nasa_prognostic_score([20.0], [10.0])
    overestimated = nasa_prognostic_score([20.0], [30.0])

    assert overestimated > underestimated


def test_regression_metrics_match_known_values_and_clip_negative_predictions() -> None:
    metrics = evaluate_regression([0.0, 10.0], [-5.0, 14.0])

    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(8.0))
    assert metrics["r2"] == pytest.approx(0.68)
    assert metrics["nasa_score"] == pytest.approx(np.exp(0.4) - 1.0)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        evaluate_regression([1.0, 2.0], [1.0])
