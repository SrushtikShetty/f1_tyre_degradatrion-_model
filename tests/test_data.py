import pandas as pd

from f1_tyre.data import build_target_frame
from models.common import create_future_target


def test_build_target_frame_creates_delta_target():
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

    out = build_target_frame(df)

    assert "next_lap_wear_increment" in out.columns
    assert out["next_lap_wear_increment"].tolist() == [2.0, 3.0, 3.0]
    assert "future_tire_wear_pct" in out.columns
    assert out["future_tire_wear_pct"].tolist() == [12.0, 15.0, 18.0]
    assert len(out) == 3


def test_build_target_frame_requires_required_columns():
    df = pd.DataFrame({"race_id": [1], "driver_id": [10]})

    try:
        build_target_frame(df)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


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
