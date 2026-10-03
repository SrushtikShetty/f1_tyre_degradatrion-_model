from __future__ import annotations

import re
from typing import Iterable, Sequence


ALLOWED_FEATURES = {
    # group keys used only for partitioning, not direct model features
    "race_id",
    "driver_id",
    "lap",
    "race_round",
    "season",
    "circuit_id",
    "circuit_length_km",
    "circuit_turns",
    "circuit_drs_zones",
    "circuit_overtake_difficulty",
    "circuit_base_lap_time_sec",
    "driver_skill_rating",
    "driver_aggression_rating",
    "driver_consistency_rating",
    "team_id",
    "team_budget_tier",
    "team_car_speed_rating",
    "team_car_downforce_rating",
    "team_engine_supplier",
    "team_car_reliability_rating",
    "team_pit_crew_rating",
    "grid_position",
    "q1_time_sec",
    "q2_time_sec",
    "q3_time_sec",
    "race_total_laps",
    "race_weather_start",
    "weather_current",
    "race_air_temp_c",
    "race_track_temp_c",
    "race_humidity_pct",
    "wind_speed_kph",
    "track_grip_level",
    "position",
    "lap_time_sec",
    "s1_time_sec",
    "s2_time_sec",
    "s3_time_sec",
    "sector_total_sec",
    "tire_compound",
    "tire_age_laps",
    "tire_age_ratio",
    "fuel_load_kg",
    "fuel_ratio",
    "ers_deploy_pct",
    "ers_harvest_pct",
    "drs_activated",
    "gap_to_leader_sec",
    "gap_ahead_sec",
    "gap_behind_sec",
    "total_nearest_gap",
    "track_status",
    "lap_time_per_km",
    "position_change",
    "race_progress",
    "car_performance_index",
    "tire_age_squared",
    "lap_time_sec_lag1",
    "lap_time_sec_lag2",
    "lap_time_sec_lag3",
    "s1_time_sec_lag1",
    "s1_time_sec_lag2",
    "s1_time_sec_lag3",
    "s2_time_sec_lag1",
    "s2_time_sec_lag2",
    "s2_time_sec_lag3",
    "s3_time_sec_lag1",
    "s3_time_sec_lag2",
    "s3_time_sec_lag3",
    "position_lag1",
    "position_lag2",
    "position_lag3",
    "gap_to_leader_sec_lag1",
    "gap_to_leader_sec_lag2",
    "gap_to_leader_sec_lag3",
    "gap_ahead_sec_lag1",
    "gap_ahead_sec_lag2",
    "gap_ahead_sec_lag3",
    "gap_behind_sec_lag1",
    "gap_behind_sec_lag2",
    "gap_behind_sec_lag3",
    "fuel_load_kg_lag1",
    "fuel_load_kg_lag2",
    "fuel_load_kg_lag3",
    "ers_deploy_pct_lag1",
    "ers_deploy_pct_lag2",
    "ers_deploy_pct_lag3",
    "ers_harvest_pct_lag1",
    "ers_harvest_pct_lag2",
    "ers_harvest_pct_lag3",
    "race_track_temp_c_lag1",
    "race_track_temp_c_lag2",
    "race_track_temp_c_lag3",
    "race_air_temp_c_lag1",
    "race_air_temp_c_lag2",
    "race_air_temp_c_lag3",
    "race_humidity_pct_lag1",
    "race_humidity_pct_lag2",
    "race_humidity_pct_lag3",
    "wind_speed_kph_lag1",
    "wind_speed_kph_lag2",
    "wind_speed_kph_lag3",
    "track_grip_level_lag1",
    "track_grip_level_lag2",
    "track_grip_level_lag3",
    "lap_time_sec_roll3_mean",
    "lap_time_sec_roll5_mean",
    "s1_time_sec_roll3_mean",
    "s1_time_sec_roll5_mean",
    "s2_time_sec_roll3_mean",
    "s2_time_sec_roll5_mean",
    "s3_time_sec_roll3_mean",
    "s3_time_sec_roll5_mean",
    "lap_time_sec_roll3_std",
    "lap_time_sec_roll5_std",
}

FORBIDDEN_FEATURES = {
    "tire_wear_pct",
    "future_tire_wear_pct",
    "next_lap_degradation_pct",
    "next_lap_wear_increment",
    "next_lap",
    "next_tire_age",
    "next_compound",
    "finish_position",
    "status",
    "points",
    "fastest_lap",
    "race_weather_end",
    "race_safety_car_deployed",
    "race_red_flag",
    "pit_stop_this_lap",
    "pit_stop_duration_sec",
    "pit_new_compound",
}

PARTITION_ONLY_FEATURES = {"race_id", "driver_id"}
CATEGORICAL_FEATURES = {
    "tire_compound",
    "track_status",
    "weather_current",
    "race_weather_start",
    "team_engine_supplier",
}
PRE_RACE_FEATURES = {
    "season",
    "race_round",
    "circuit_id",
    "circuit_length_km",
    "circuit_turns",
    "circuit_drs_zones",
    "circuit_overtake_difficulty",
    "circuit_base_lap_time_sec",
    "driver_skill_rating",
    "driver_aggression_rating",
    "driver_consistency_rating",
    "team_id",
    "team_budget_tier",
    "team_car_speed_rating",
    "team_car_downforce_rating",
    "team_engine_supplier",
    "team_car_reliability_rating",
    "team_pit_crew_rating",
    "grid_position",
    "q1_time_sec",
    "q2_time_sec",
    "q3_time_sec",
    "race_total_laps",
    "race_weather_start",
}
ENGINEERED_LINEAGE = {
    "race_progress": ("lap", "race_total_laps"),
    "tire_age_ratio": ("tire_age_laps", "race_total_laps"),
    "tire_age_squared": ("tire_age_laps",),
    "total_nearest_gap": ("gap_ahead_sec", "gap_behind_sec"),
    "position_change": ("grid_position", "position"),
    "lap_time_per_km": ("lap_time_sec", "circuit_length_km"),
    "sector_total_sec": ("s1_time_sec", "s2_time_sec", "s3_time_sec"),
}


def _metadata_for(feature: str) -> dict[str, object]:
    lag_match = re.fullmatch(r"(.+)_lag(\d+)", feature)
    rolling_match = re.fullmatch(r"(.+)_roll(\d+)_(mean|std)", feature)
    if lag_match:
        source = lag_match.group(1)
        lineage = (source,)
        availability = "previous_lap_history"
        transformation = f"shift({lag_match.group(2)}) within race-driver order"
        derived, lagged, rolling = True, True, False
    elif rolling_match:
        source = rolling_match.group(1)
        lineage = (source,)
        availability = "rolling_prior_history"
        transformation = f"shift(1), then rolling {rolling_match.group(2)} {rolling_match.group(3)}"
        derived, lagged, rolling = True, False, True
    elif feature in ENGINEERED_LINEAGE:
        lineage = ENGINEERED_LINEAGE[feature]
        source = ", ".join(lineage)
        availability = "derived_current_state"
        transformation = {
            "race_progress": "lap / scheduled race_total_laps",
            "tire_age_ratio": "tire_age_laps / supplied race_total_laps",
            "tire_age_squared": "tire_age_laps ** 2",
            "total_nearest_gap": "abs(gap_ahead_sec) + abs(gap_behind_sec)",
            "position_change": "grid_position - position",
            "lap_time_per_km": "lap_time_sec / circuit_length_km",
            "sector_total_sec": "sum(current sector times)",
        }[feature]
        derived, lagged, rolling = True, False, False
    else:
        source = feature
        lineage = (feature,)
        availability = (
            "partition_only" if feature in PARTITION_ONLY_FEATURES
            else "pre_race_or_known_by_prediction_time" if feature in PRE_RACE_FEATURES
            else "current_lap_observation"
        )
        transformation = "identity"
        derived, lagged, rolling = False, False, False

    is_forbidden = feature in FORBIDDEN_FEATURES or any(pattern in feature.lower() for pattern in (
        "future_", "next_", "finish_position", "race_end", "race_result", "fastest_lap", "pit_stop_", "pit_new_"
    ))
    return {
        "name": feature,
        "source": source,
        "availability": availability,
        "data_type": "categorical" if feature in CATEGORICAL_FEATURES else "numeric_or_identifier",
        "transformation": transformation,
        "derived": derived,
        "lagged": lagged,
        "rolling": rolling,
        "allowed": feature in ALLOWED_FEATURES and feature not in PARTITION_ONLY_FEATURES and not is_forbidden,
        "lineage": lineage,
    }


FEATURE_METADATA = {
    feature: _metadata_for(feature)
    for feature in ALLOWED_FEATURES | FORBIDDEN_FEATURES | PARTITION_ONLY_FEATURES
}


def validate_feature_names(features: Iterable[str] | Sequence[str]) -> list[str]:
    names = [str(feature) for feature in features]
    unknown = [name for name in names if name not in ALLOWED_FEATURES]
    forbidden = [name for name in names if name in FORBIDDEN_FEATURES]
    partition_only = [name for name in names if name in PARTITION_ONLY_FEATURES]

    if forbidden:
        raise ValueError(f"Forbidden leakage features detected: {sorted(set(forbidden))}")
    if partition_only:
        raise ValueError(f"Partition identifiers cannot be model features: {sorted(set(partition_only))}")
    if unknown:
        raise ValueError(
            "Unknown feature names are not allowed by the causal policy: "
            + ", ".join(sorted(set(unknown)))
        )
    return names
