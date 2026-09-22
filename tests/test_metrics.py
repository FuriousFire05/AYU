import pytest

from ayu.metrics import nasa_prognostic_score


def test_nasa_score_is_zero_for_exact_predictions() -> None:
    assert nasa_prognostic_score([20.0], [20.0]) == pytest.approx(0.0)


def test_nasa_score_penalizes_overestimation_more() -> None:
    underestimated = nasa_prognostic_score([20.0], [10.0])
    overestimated = nasa_prognostic_score([20.0], [30.0])

    assert overestimated > underestimated
