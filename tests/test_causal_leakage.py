import pandas as pd
import pytest

from f1_tyre.evaluation.leakage_audit import audit_feature_matrix
from f1_tyre.features import add_shifted_rolling
from f1_tyre.strict_feature_policy import FEATURE_METADATA, validate_feature_names


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


@pytest.mark.parametrize(
    "column",
    ["next_lap_degradation_pct", "next_lap_wear_increment", "future_tire_wear_pct", "tire_wear_pct"],
)
def test_feature_audit_rejects_generated_target_and_future_state(column):
    with pytest.raises(ValueError):
        audit_feature_matrix(pd.DataFrame({column: [1.0]}))


@pytest.mark.parametrize(
    "column",
    ["future_surface_summary", "next_lap_lap_time_sec", "next_tyre_state_proxy", "race_result_position"],
)
def test_feature_audit_rejects_pattern_based_future_or_result_proxies(column):
    with pytest.raises(ValueError, match="Leakage detected"):
        audit_feature_matrix(pd.DataFrame({column: [1.0]}))


def test_policy_rejects_unknown_feature_names():
    with pytest.raises(ValueError):
        validate_feature_names(["race_id", "driver_id", "lap", "unknown_feature"])


def test_registry_covers_allowlisted_features_with_required_metadata():
    required = {
        "source",
        "availability",
        "data_type",
        "transformation",
        "derived",
        "lagged",
        "rolling",
        "allowed",
        "lineage",
    }

    assert FEATURE_METADATA
    assert all(required <= set(metadata) for metadata in FEATURE_METADATA.values())
    assert FEATURE_METADATA["lap_time_sec_roll5_mean"]["rolling"] is True
    assert FEATURE_METADATA["lap_time_sec_roll5_mean"]["availability"] == "rolling_prior_history"


def test_current_track_status_is_not_mistaken_for_race_result_status():
    assert audit_feature_matrix(pd.DataFrame({"track_status": ["GREEN"]})) is True


def test_race_result_status_column_is_rejected():
    with pytest.raises(ValueError, match="Leakage detected"):
        audit_feature_matrix(pd.DataFrame({"status": ["Finished"]}))


def test_partition_identifiers_cannot_be_model_features():
    with pytest.raises(ValueError, match="Partition identifiers"):
        audit_feature_matrix(pd.DataFrame({"race_id": [1]}))


def test_lineage_audit_rejects_future_source_under_allowed_feature_name():
    features = pd.DataFrame({"lap_time_sec": [90.0]})

    with pytest.raises(ValueError, match="feature lineage"):
        audit_feature_matrix(
            features,
            lineage={"lap_time_sec": ["next_lap_lap_time_sec"]},
        )


def test_lineage_audit_rejects_unregistered_source():
    features = pd.DataFrame({"lap_time_sec": [90.0]})

    with pytest.raises(ValueError, match="feature lineage"):
        audit_feature_matrix(
            features,
            lineage={"lap_time_sec": ["future_surface_summary"]},
        )


def test_lineage_audit_rejects_partition_key_source():
    features = pd.DataFrame({"lap_time_sec": [90.0]})

    with pytest.raises(ValueError, match="feature lineage"):
        audit_feature_matrix(
            features,
            lineage={"lap_time_sec": ["race_id"]},
        )


def test_allowed_feature_lineage_sources_are_registered_and_available():
    errors = []
    for feature, metadata in FEATURE_METADATA.items():
        if not metadata["allowed"]:
            continue
        for source in metadata["lineage"]:
            source_metadata = FEATURE_METADATA.get(source)
            if source_metadata is None:
                errors.append(f"{feature} <- {source}: missing source metadata")
            elif not source_metadata["allowed"]:
                errors.append(f"{feature} <- {source}: source is not model-available")

    assert errors == []


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
