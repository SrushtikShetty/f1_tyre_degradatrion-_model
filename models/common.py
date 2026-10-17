from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.f1_tyre.config import PROJECT_ROOT
from src.f1_tyre.data import build_legacy_target_frame
from src.f1_tyre.features import add_common_engineered_features
from src.f1_tyre.model_loader import save_model_artifact
from src.f1_tyre.metrics import compute_metrics

RANDOM_STATE = 42
TARGET = "next_lap_wear_increment"
ALLOWED_RAW_FEATURES = {
    "season", "race_round", "circuit_id", "circuit_length_km", "circuit_turns",
    "circuit_drs_zones", "circuit_overtake_difficulty", "circuit_base_lap_time_sec",
    "driver_skill_rating", "driver_aggression_rating", "driver_consistency_rating",
    "team_id", "team_budget_tier", "team_car_speed_rating", "team_car_downforce_rating",
    "team_engine_supplier", "team_car_reliability_rating", "team_pit_crew_rating",
    "grid_position", "q1_time_sec", "q2_time_sec", "q3_time_sec", "race_total_laps",
    "race_weather_start", "lap", "position", "lap_time_sec", "s1_time_sec", "s2_time_sec",
    "s3_time_sec", "tire_compound", "tire_age_laps", "fuel_load_kg", "ers_deploy_pct",
    "ers_harvest_pct", "drs_activated", "gap_to_leader_sec", "gap_ahead_sec",
    "gap_behind_sec", "track_status", "weather_current", "race_air_temp_c",
    "race_track_temp_c", "race_humidity_pct",
}
FORBIDDEN_FEATURES = {
    "tire_wear_pct", "future_tire_wear_pct", "next_lap_wear_increment", "next_lap",
    "next_tire_age", "next_compound", "finish_position", "status", "points", "fastest_lap",
    "race_weather_end", "race_safety_car_deployed", "race_red_flag", "pit_stop_this_lap",
    "pit_stop_duration_sec", "pit_new_compound", "race_id", "driver_id",
}
ENGINEERED = [
    "sector_total_sec", "lap_time_per_km", "tire_age_ratio", "aggression_x_tire_age",
    "consistency_x_tire_age", "track_temp_x_tire_age", "air_temp_x_tire_age",
    "overtake_difficulty_x_tire_age", "fuel_ratio", "total_nearest_gap", "position_change",
    "race_progress", "car_performance_index",
]


def label_supplementary_grouped_cv(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        **metrics,
        "metric_scope": "supplementary_grouped_cv",
        "metric_protocol": "5-fold GroupKFold grouped by race_id using out-of-fold predictions",
        "primary_metric": False,
    }


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    boolean_columns = cleaned.select_dtypes(include=["bool"]).columns
    cleaned[boolean_columns] = cleaned[boolean_columns].astype(int)
    empty_columns = cleaned.columns[cleaned.isna().all()]
    return cleaned.drop(columns=empty_columns)


def create_future_target(train: pd.DataFrame) -> pd.DataFrame:
    frame = build_legacy_target_frame(train)
    return frame.loc[np.isfinite(frame["future_tire_wear_pct"])].copy()


def create_current_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = add_common_engineered_features(frame)
    columns = set(features.columns)
    tire_age = features.get("tire_age_laps")
    if tire_age is not None:
        for source, target in (
            ("driver_aggression_rating", "aggression_x_tire_age"),
            ("driver_consistency_rating", "consistency_x_tire_age"),
            ("race_track_temp_c", "track_temp_x_tire_age"),
            ("race_air_temp_c", "air_temp_x_tire_age"),
            ("circuit_overtake_difficulty", "overtake_difficulty_x_tire_age"),
        ):
            if source in features:
                features[target] = features[source] * tire_age
    if {"fuel_load_kg", "race_total_laps"} <= columns:
        features["fuel_ratio"] = features["fuel_load_kg"] / features["race_total_laps"].replace(0, np.nan)
    if {"team_car_speed_rating", "team_car_downforce_rating"} <= columns:
        features["car_performance_index"] = features["team_car_speed_rating"] * features["team_car_downforce_rating"]
    return features


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [name for name in ALLOWED_RAW_FEATURES if name in frame.columns]
    columns.extend(name for name in ENGINEERED if name in frame.columns)
    columns = list(dict.fromkeys(columns))
    forbidden = set(columns).intersection(FORBIDDEN_FEATURES)
    if forbidden:
        raise ValueError(f"Forbidden features: {sorted(forbidden)}")
    return columns


def prepare_dataset(
    data_dir: str | Path = ".",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, list[str]]:
    root = Path(data_dir)
    train = clean_data(pd.read_csv(root / "train.csv"))
    test = pd.read_csv(root / "test.csv")
    if TARGET in test:
        test = test.drop(columns=[TARGET])
    train = create_current_features(create_future_target(train))
    test = create_current_features(clean_data(test))
    columns = feature_columns(train)
    features = train[columns].copy()
    target = train[TARGET].astype(float)
    groups = train["race_id"].copy()

    for column in columns:
        if column not in test:
            test[column] = np.nan
    return features, target, groups, test, columns


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), categorical),
        ]
    )


def train_grouped_cv_model(
    model_name: str,
    model_label: str,
    estimator: BaseEstimator,
    data_dir: str | Path = ".",
    out_dir: str | Path = "artifacts",
    *,
    cv_jobs: int = 1,
    scale: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features, target, groups, _, columns = prepare_dataset(data_dir)
    steps = [("prep", build_preprocessor(features))]
    if scale:
        steps.append(("scale", StandardScaler(with_mean=False)))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)

    predictions = cross_val_predict(
        pipeline,
        features,
        target,
        cv=GroupKFold(n_splits=5),
        groups=groups,
        n_jobs=cv_jobs,
    )
    metrics = label_supplementary_grouped_cv(
        {"model": model_label, **compute_metrics(target, predictions), **(metadata or {})}
    )

    pipeline.fit(features, target)
    artifact = {
        "model": pipeline,
        "feature_columns": columns,
        "target": TARGET,
        "metrics": metrics,
        "metadata": {"model_name": model_name, **(metadata or {})},
    }
    save_model_artifact(model_name, artifact, PROJECT_ROOT / out_dir / "models")
    return metrics
