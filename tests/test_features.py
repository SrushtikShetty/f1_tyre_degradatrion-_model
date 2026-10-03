import pandas as pd

from f1_tyre.features import add_group_lags, add_shifted_rolling, add_engineered_features, prepare_feature_matrix


def test_add_group_lags_and_rolling_features():
    df = pd.DataFrame(
        {
            "race_id": [1, 1, 1, 1],
            "driver_id": [1, 1, 1, 1],
            "lap": [1, 2, 3, 4],
            "race_total_laps": [4, 4, 4, 4],
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


def test_feature_row_is_invariant_to_changes_in_future_telemetry():
    df = pd.DataFrame(
        {
            "race_id": [1] * 5,
            "driver_id": [10] * 5,
            "lap": [1, 2, 3, 4, 5],
            "race_total_laps": [5] * 5,
            "tire_age_laps": [1, 2, 3, 4, 5],
            "lap_time_sec": [90.0, 92.0, 91.0, 94.0, 95.0],
            "s1_time_sec": [30.0, 31.0, 30.5, 32.0, 33.0],
            "position": [3, 3, 2, 2, 1],
            "race_track_temp_c": [30.0, 31.0, 32.0, 33.0, 34.0],
        }
    )
    changed_future = df.copy()
    changed_future.loc[4, ["lap_time_sec", "s1_time_sec", "position", "race_track_temp_c"]] = [
        999.0,
        333.0,
        9,
        80.0,
    ]

    baseline = prepare_feature_matrix(df).loc[2]
    changed = prepare_feature_matrix(changed_future).loc[2]

    pd.testing.assert_series_equal(baseline, changed)


def test_age_ratio_is_not_inferred_from_observed_race_maximum():
    df = pd.DataFrame(
        {
            "race_id": [1, 1, 1],
            "driver_id": [10, 10, 10],
            "lap": [1, 2, 3],
            "tire_age_laps": [1, 2, 3],
            "lap_time_sec": [90.0, 92.0, 91.0],
        }
    )

    features = prepare_feature_matrix(df)

    assert "tire_age_ratio" not in features.columns


def test_feature_preparation_sorts_before_lag_and_rolling_calculation():
    df = pd.DataFrame(
        {
            "race_id": [1, 1, 1],
            "driver_id": [10, 10, 10],
            "lap": [3, 1, 2],
            "lap_time_sec": [30.0, 10.0, 20.0],
        }
    )

    features = prepare_feature_matrix(df)

    assert features["lap"].tolist() == [1, 2, 3]
    assert features["lap_time_sec_lag1"].iloc[2] == 20.0
    assert features["lap_time_sec_roll3_mean"].iloc[2] == 15.0
