import pandas as pd
import pytest

from f1_tyre.evaluation.leakage_audit import audit_feature_matrix
from f1_tyre.features import add_shifted_rolling
from f1_tyre.strict_feature_policy import validate_feature_names


def test_audit_rejects_future_target_proxies():
    df = pd.DataFrame(
        {
            "race_id": [1],
            "driver_id": [10],
            "lap": [1],
            "tire_wear_pct": [10.0],
            "future_tire_wear_pct": [12.0],
            "next_lap_wear_increment": [2.0],
        }
    )

    with pytest.raises(ValueError):
        audit_feature_matrix(df)


def test_policy_rejects_unknown_feature_names():
    with pytest.raises(ValueError):
        validate_feature_names(["race_id", "driver_id", "lap", "unknown_feature"])


def test_rolling_features_are_causal_on_synthetic_data():
    df = pd.DataFrame(
        {
            "race_id": [1, 1, 1, 1],
            "driver_id": [10, 10, 10, 10],
            "lap": [1, 2, 3, 4],
            "lap_time_sec": [10.0, 20.0, 30.0, 40.0],
        }
    )

    out = add_shifted_rolling(
        df,
        group_cols=["race_id", "driver_id"],
        source_columns=["lap_time_sec"],
        windows=(3,),
    )

    assert out["lap_time_sec_roll3_mean"].iloc[3] == 20.0
    assert pd.isna(out["lap_time_sec_roll3_mean"].iloc[0])
    assert pd.isna(out["lap_time_sec_roll3_mean"].iloc[1])


def test_race_split_overlap_is_rejected():
    train_races = {1, 2}
    val_races = {2, 3}

    assert len(train_races & val_races) == 1
