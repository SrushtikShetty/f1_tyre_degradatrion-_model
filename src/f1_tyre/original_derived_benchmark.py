from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TARGET_NAME = "next_lap_wear_increment"
LEGACY_TARGET_COLUMNS = {
    "future_tire_wear_pct",
    "next_lap_wear_increment",
    "next_lap_degradation_pct",
}

_COMPOUND_BASE = {"HARD": 0.28, "MEDIUM": 0.42, "SOFT": 0.58}
_COMPOUND_OPTIMAL_TEMP = {"HARD": 35.0, "MEDIUM": 38.0, "SOFT": 40.0}


def _as_float(series: pd.Series | None, default: float = 0.0, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(np.full(0, default), dtype=float)
        return pd.Series(np.full(len(index), default, dtype=float), index=index, dtype=float)
    values = pd.to_numeric(series, errors="coerce").fillna(default)
    return values.astype(float)


def _guard_legacy_target_inputs(frame: pd.DataFrame) -> None:
    forbidden = sorted(
        {column for column in LEGACY_TARGET_COLUMNS if column in frame.columns}
        | {
            column
            for column in frame.columns
            if str(column).lower().startswith(("next_", "future_"))
            and column not in {"next_lap_wear_increment", "next_lap_degradation_pct"}
        }
    )
    if forbidden:
        raise ValueError(
            "Legacy target columns or future-state columns cannot be used to generate the original-data-derived target: "
            + ", ".join(forbidden)
        )


def _compound_factor(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.upper().str.strip()
    return values.map(_COMPOUND_BASE).fillna(0.4)


def _temperature_term(track_temp: pd.Series, compound: pd.Series) -> pd.Series:
    optimum = compound.map(_COMPOUND_OPTIMAL_TEMP).fillna(38.0)
    gap = (track_temp - optimum).abs() / 30.0
    return 1.0 + 0.06 * gap


def _build_semisynthetic_target(frame: pd.DataFrame, seed: int = 42) -> pd.Series:
    _guard_legacy_target_inputs(frame)

    rng = np.random.default_rng(seed)
    n_rows = len(frame)
    compound = frame.get("tire_compound", pd.Series(["HARD"] * n_rows, index=frame.index)).astype(str).str.upper().str.strip()
    age = _as_float(frame.get("tire_age_laps"), default=0.0, index=frame.index)
    lap_time = _as_float(frame.get("lap_time_sec"), default=90.0, index=frame.index)
    circuit_length = _as_float(frame.get("circuit_length_km"), default=5.0, index=frame.index)
    circuit_turns = _as_float(frame.get("circuit_turns"), default=18.0, index=frame.index)
    track_temp = _as_float(frame.get("race_track_temp_c"), default=35.0, index=frame.index)
    air_temp = _as_float(frame.get("race_air_temp_c"), default=25.0, index=frame.index)
    fuel_load = _as_float(frame.get("fuel_load_kg"), default=90.0, index=frame.index)
    aggress = _as_float(frame.get("driver_aggression_rating"), default=75.0, index=frame.index) / 100.0
    skills = _as_float(frame.get("driver_skill_rating"), default=80.0, index=frame.index) / 100.0
    consistency = _as_float(frame.get("driver_consistency_rating"), default=80.0, index=frame.index) / 100.0
    humidity = _as_float(frame.get("race_humidity_pct"), default=45.0, index=frame.index)
    wind = _as_float(frame.get("wind_speed_kph"), default=5.0, index=frame.index)
    grip = _as_float(frame.get("track_grip_level"), default=0.9, index=frame.index)
    gap_ahead = _as_float(frame.get("gap_ahead_sec"), default=1.0, index=frame.index)
    gap_behind = _as_float(frame.get("gap_behind_sec"), default=1.0, index=frame.index)
    lap = _as_float(frame.get("lap"), default=1.0, index=frame.index)
    total_laps = _as_float(frame.get("race_total_laps"), default=1.0, index=frame.index)
    fuel_ratio = _as_float(frame.get("fuel_load_kg"), default=90.0, index=frame.index) / 100.0
    progress = (lap / total_laps.replace(0, 1)).clip(lower=0.0, upper=1.0)
    circuit_load = (circuit_turns / 30.0 + (circuit_length / 8.0) / 2.0).clip(lower=0.0, upper=1.5)
    pace_term = 0.8 + 0.18 * (lap_time / (circuit_length * 20.0 + 1e-6)).clip(lower=0.0, upper=1.5)
    age_term = 1.0 + 0.025 * age + 0.0006 * age.pow(2)
    thermal_term = _temperature_term(track_temp, compound)
    circuit_term = 0.9 + 0.22 * (circuit_load / 1.5)
    fuel_term = 0.9 + 0.22 * fuel_ratio
    driver_term = 0.9 + 0.12 * aggress + 0.08 * skills + 0.06 * consistency
    traffic_term = 0.96 + 0.06 * ((gap_ahead + gap_behind) / 10.0).clip(lower=0.0, upper=1.0)
    weather_term = 1.0 + 0.012 * (humidity / 100.0) + 0.01 * (wind / 20.0) + 0.04 * (1.0 - grip)
    phase_term = 0.9 + 0.22 * progress
    latent = 0.92 + 0.18 * rng.random(n_rows)
    target = (
        _compound_factor(compound)
        * age_term
        * thermal_term
        * circuit_term
        * pace_term
        * fuel_term
        * driver_term
        * traffic_term
        * weather_term
        * phase_term
        * latent
    )
    return target.clip(lower=0.0, upper=8.0)


def build_original_derived_dataset(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    frames = []
    for source_name, source_frame in (("train.csv", train), ("test.csv", test)):
        if source_frame is None:
            continue
        row_frame = source_frame.copy().reset_index(drop=True)
        row_frame["_source_file"] = source_name
        row_frame["_source_row_number"] = np.arange(len(row_frame))
        row_frame[TARGET_NAME] = _build_semisynthetic_target(row_frame, seed=seed + len(frames) * 17)
        frames.append(row_frame)
    if not frames:
        raise ValueError("At least one source dataset must be provided.")
    derived = pd.concat(frames, ignore_index=True)
    drop_columns = [
        column
        for column in derived.columns
        if (column.startswith("next_") or column.startswith("future_"))
        and column not in {TARGET_NAME}
    ]
    if drop_columns:
        derived = derived.drop(columns=drop_columns, errors="ignore")
    return derived


def build_original_derived_manifest(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    train_count = 0 if train is None else int(len(train))
    test_count = 0 if test is None else int(len(test))
    derived = build_original_derived_dataset(train, test, seed=seed)
    return {
        "source_files": ["train.csv", "test.csv"],
        "source_row_counts": {"train.csv": train_count, "test.csv": test_count},
        "source_total_rows": train_count + test_count,
        "derived_row_count": int(len(derived)),
        "lineage_columns": ["_source_file", "_source_row_number"],
        "target_name": TARGET_NAME,
        "target_definition": "simulated tyre-wear increment for the next lap from current-lap observable state only",
        "seed": int(seed),
        "dataset_hash": "sha256:pending",
        "notes": "The target is derived from current observable race-state columns and does not use tire_wear_pct or any future-state column.",
    }


__all__ = ["TARGET_NAME", "build_original_derived_dataset", "build_original_derived_manifest"]
