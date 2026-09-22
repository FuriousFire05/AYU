import pandas as pd
import pytest

from ayu.features import add_causal_rolling_features, determine_informative_sensors


def test_rolling_features_reset_at_engine_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "sensor_1": [1.0, 3.0, 10.0, 14.0],
        }
    )

    result = add_causal_rolling_features(frame, ["sensor_1"], windows=(5,))

    assert result["sensor_1_rolling_mean_5"].tolist() == [1.0, 2.0, 10.0, 12.0]
    assert result.loc[2, "sensor_1_rolling_std_5"] == 0.0


def test_future_value_cannot_change_earlier_rolling_feature() -> None:
    original = pd.DataFrame(
        {
            "unit_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "sensor_1": [2.0, 4.0, 6.0],
        }
    )
    changed = original.copy()
    changed.loc[2, "sensor_1"] = 6_000.0

    original_features = add_causal_rolling_features(
        original, ["sensor_1"], windows=(5,)
    )
    changed_features = add_causal_rolling_features(changed, ["sensor_1"], windows=(5,))

    assert changed_features.loc[1, "sensor_1_rolling_mean_5"] == pytest.approx(
        original_features.loc[1, "sensor_1_rolling_mean_5"]
    )


def test_constant_sensor_is_excluded() -> None:
    frame = pd.DataFrame(
        {
            "sensor_constant": [1.0, 1.0, 1.0],
            "sensor_variable": [1.0, 2.0, 3.0],
        }
    )

    selected = determine_informative_sensors(
        frame, ["sensor_constant", "sensor_variable"]
    )

    assert selected == ["sensor_variable"]
