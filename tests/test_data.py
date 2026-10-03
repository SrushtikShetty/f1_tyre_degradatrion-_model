import pandas as pd
import pytest

from f1_tyre.data import build_legacy_target_frame, build_model_features, build_target_frame
from models.common import create_future_target


def test_build_target_frame_uses_explicit_simulated_target_without_future_rows():
    df = pd.DataFrame(
        {
            "race_id": [1, 1, 1, 1],
            "driver_id": [10, 10, 10, 10],
            "lap": [1, 2, 3, 4],
            "tire_age_laps": [0, 1, 2, 3],
            "tire_compound": ["C1", "C1", "C1", "C1"],
            "tire_wear_pct": [0.0, 1.2, 2.5, 3.7],
            "next_lap_degradation_pct": [1.2, 1.3, 1.2, float("nan")],
            "next_lap_lap_time_sec": [91.0, 92.0, 93.0, 94.0],
        }
    )

    out = build_target_frame(df)

    assert "next_lap_wear_increment" in out.columns
    assert out["next_lap_wear_increment"].tolist() == [1.2, 1.3, 1.2]
    assert "next_lap_degradation_pct" not in out.columns
    assert "next_lap_lap_time_sec" not in out.columns
    assert "tire_wear_pct" not in out.columns
    assert len(out) == 3


def test_legacy_target_shift_is_explicitly_separate():
    df = pd.DataFrame(
        {
            "race_id": [1, 1, 1],
            "driver_id": [10, 10, 10],
            "lap": [1, 2, 3],
            "tire_age_laps": [1, 2, 3],
            "tire_compound": ["HARD", "HARD", "HARD"],
            "tire_wear_pct": [0.0, 1.0, 2.0],
        }
    )

    legacy = build_legacy_target_frame(df)

    assert legacy["next_lap_wear_increment"].tolist() == [1.0, 1.0]
    assert legacy["future_tire_wear_pct"].tolist() == [1.0, 2.0]


def test_build_target_frame_requires_required_columns():
    df = pd.DataFrame({"race_id": [1], "driver_id": [10]})

    try:
        build_target_frame(df)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


@pytest.mark.parametrize(
    "column",
    ["next_lap_degradation_pct", "next_lap_wear_increment", "future_tire_wear_pct", "tire_wear_pct"],
)
def test_target_or_future_columns_cannot_be_selected_as_features(column):
    df = pd.DataFrame({"lap": [1], "position": [2], column: [1.0]})

    with pytest.raises(ValueError, match="cannot enter X"):
        build_model_features(df, [column])


def test_build_model_features_accepts_allowlisted_current_state():
    df = pd.DataFrame({"lap": [1], "position": [2]})

    assert list(build_model_features(df, ["lap", "position"]).columns) == ["lap", "position"]


def test_multi_model_target_is_next_lap_wear_increment():
    df = pd.DataFrame(
        {
            "race_id": [1, 1, 1, 1],
            "driver_id": [10, 10, 10, 10],
            "lap": [1, 2, 3, 4],
            "tire_age_laps": [0, 1, 2, 3],
            "tire_compound": ["C1", "C1", "C1", "C1"],
            "tire_wear_pct": [10.0, 12.0, 15.0, 18.0],
        }
    )

    out = create_future_target(df)

    assert out["next_lap_wear_increment"].tolist() == [2.0, 3.0, 3.0]
    assert out["future_tire_wear_pct"].tolist() == [12.0, 15.0, 18.0]
