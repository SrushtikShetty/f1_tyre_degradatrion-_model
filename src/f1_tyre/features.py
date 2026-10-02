from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COMMON_GROUP_COLS


def add_group_lags(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    source_columns: list[str] | None = None,
    lags: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    out = df.copy()
    group_cols = group_cols or COMMON_GROUP_COLS
    source_columns = source_columns or []
    for column in source_columns:
        if column not in out.columns:
            continue
        grouped = out.groupby(group_cols, sort=False)[column]
        for lag in lags:
            out[f"{column}_lag{lag}"] = grouped.shift(lag)
    return out


def add_shifted_rolling(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    source_columns: list[str] | None = None,
    windows: tuple[int, ...] = (3, 5),
) -> pd.DataFrame:
    out = df.copy()
    group_cols = group_cols or COMMON_GROUP_COLS
    source_columns = source_columns or []
    for column in source_columns:
        if column not in out.columns:
            continue
        shifted = out.groupby(group_cols, sort=False)[column].shift(1)
        groups = [out[group] for group in group_cols]
        for window in windows:
            rolling = shifted.groupby(groups, sort=False).rolling(window=window, min_periods=max(2, min(2, window)))
            out[f"{column}_roll{window}_mean"] = rolling.mean().reset_index(level=list(range(len(group_cols))), drop=True)
            if window >= 5:
                out[f"{column}_roll{window}_std"] = rolling.std().reset_index(level=list(range(len(group_cols))), drop=True)
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"lap", "race_total_laps"}.issubset(out.columns):
        out["race_progress"] = out["lap"] / out["race_total_laps"].replace(0, np.nan)
    if "tire_age_laps" in out.columns:
        if "race_total_laps" in out.columns:
            denominator = out["race_total_laps"].replace(0, np.nan)
        else:
            denominator = out.groupby("race_id")["lap"].transform("max").replace(0, np.nan)
        out["tire_age_ratio"] = out["tire_age_laps"] / denominator
        out["tire_age_squared"] = out["tire_age_laps"] ** 2
    if {"gap_ahead_sec", "gap_behind_sec"}.issubset(out.columns):
        out["total_nearest_gap"] = out["gap_ahead_sec"].abs() + out["gap_behind_sec"].abs()
    if {"grid_position", "position"}.issubset(out.columns):
        out["position_change"] = out["grid_position"] - out["position"]
    if {"lap_time_sec", "circuit_length_km"}.issubset(out.columns):
        out["lap_time_per_km"] = out["lap_time_sec"] / out["circuit_length_km"].replace(0, np.nan)
    if {"s1_time_sec", "s2_time_sec", "s3_time_sec"}.issubset(out.columns):
        out["sector_total_sec"] = out[["s1_time_sec", "s2_time_sec", "s3_time_sec"]].sum(axis=1)
    return out


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_group_lags(out, COMMON_GROUP_COLS, [
        "lap_time_sec", "s1_time_sec", "s2_time_sec", "s3_time_sec",
        "position", "gap_to_leader_sec", "gap_ahead_sec", "gap_behind_sec",
        "fuel_load_kg", "ers_deploy_pct", "ers_harvest_pct",
        "race_track_temp_c", "race_air_temp_c", "race_humidity_pct",
        "wind_speed_kph", "track_grip_level",
    ], lags=(1, 2, 3))
    out = add_shifted_rolling(out, COMMON_GROUP_COLS, [
        "lap_time_sec", "s1_time_sec", "s2_time_sec", "s3_time_sec",
    ], windows=(3, 5))
    out = add_engineered_features(out)
    return out
