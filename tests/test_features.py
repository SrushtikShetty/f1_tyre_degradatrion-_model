import pandas as pd

from f1_tyre.features import add_group_lags, add_shifted_rolling, add_engineered_features


def test_add_group_lags_and_rolling_features():
    df = pd.DataFrame(
        {
            "race_id": [1, 1, 1, 1],
            "driver_id": [1, 1, 1, 1],
            "lap": [1, 2, 3, 4],
            "lap_time_sec": [90.0, 92.0, 91.0, 94.0],
            "fuel_load_kg": [70.0, 69.0, 68.0, 67.0],
            "position": [2, 2, 3, 3],
            "tire_age_laps": [0, 1, 2, 3],
        }
    )

    out = add_group_lags(df, ["race_id", "driver_id"], ["lap_time_sec", "fuel_load_kg", "position"], lags=(1, 2))
    out = add_shifted_rolling(out, ["race_id", "driver_id"], ["lap_time_sec"], windows=(3,))
    out = add_engineered_features(out)

    assert "lap_time_sec_lag1" in out.columns
    assert "fuel_load_kg_lag2" in out.columns
    assert "lap_time_sec_roll3_mean" in out.columns
    assert "tire_age_ratio" in out.columns
