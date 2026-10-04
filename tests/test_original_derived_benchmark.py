from pathlib import Path

import numpy as np
import pandas as pd

from data_generation.build_original_derived_benchmark import load_split_benchmark
from f1_tyre.original_derived_benchmark import build_original_derived_dataset


def _train_rows():
    return pd.DataFrame(
        {
            "race_id": [1, 1, 1, 2, 2, 2, 3, 3],
            "driver_id": [10, 10, 10, 11, 11, 11, 12, 12],
            "season": [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024],
            "race_round": [1, 1, 1, 2, 2, 2, 3, 3],
            "lap": [1, 2, 3, 1, 2, 3, 2, 3],
            "tire_compound": ["HARD", "HARD", "HARD", "SOFT", "SOFT", "SOFT", "HARD", "HARD"],
            "tire_age_laps": [1, 2, 3, 1, 2, 3, 2, 3],
            "lap_time_sec": [90.0, 92.0, 94.0, 96.0, 99.0, 101.0, 96.0, 99.5],
            "s1_time_sec": [29.0, 30.0, 31.0, 31.0, 32.0, 33.0, 31.0, 32.5],
            "s2_time_sec": [30.0, 31.0, 31.5, 32.0, 33.0, 34.0, 32.0, 33.0],
            "s3_time_sec": [31.0, 31.0, 31.5, 33.0, 34.0, 34.0, 33.0, 34.0],
            "fuel_load_kg": [100.0, 95.0, 90.0, 100.0, 94.0, 88.0, 90.0, 84.0],
            "ers_deploy_pct": [30.0, 35.0, 40.0, 35.0, 42.0, 48.0, 38.0, 45.0],
            "ers_harvest_pct": [20.0, 22.0, 24.0, 15.0, 18.0, 19.0, 21.0, 23.0],
            "driver_aggression_rating": [70.0, 72.0, 74.0, 78.0, 80.0, 82.0, 76.0, 79.0],
            "driver_consistency_rating": [80.0, 82.0, 84.0, 75.0, 77.0, 79.0, 81.0, 83.0],
            "driver_skill_rating": [85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 86.0, 88.0],
            "race_air_temp_c": [25.0, 26.0, 27.0, 30.0, 31.0, 32.0, 27.0, 28.5],
            "race_track_temp_c": [34.0, 35.0, 37.0, 36.0, 38.0, 40.0, 35.5, 36.8],
            "race_humidity_pct": [40.0, 42.0, 44.0, 45.0, 47.0, 50.0, 41.0, 43.0],
            "wind_speed_kph": [4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 7.0, 9.0],
            "track_grip_level": [0.9, 0.88, 0.87, 0.84, 0.82, 0.8, 0.87, 0.85],
            "gap_ahead_sec": [2.0, 1.0, 0.5, 4.0, 3.0, 2.0, 1.5, 0.7],
            "gap_behind_sec": [1.0, 2.0, 1.5, 2.0, 1.5, 1.0, 1.7, 1.1],
            "gap_to_leader_sec": [1.5, 1.2, 0.8, 6.0, 5.0, 4.0, 2.0, 1.0],
            "position": [1, 1, 2, 3, 2, 1, 2, 1],
            "grid_position": [1, 1, 1, 3, 3, 3, 2, 2],
            "race_total_laps": [10, 10, 10, 10, 10, 10, 10, 10],
            "track_status": ["GREEN", "GREEN", "GREEN", "GREEN", "GREEN", "GREEN", "GREEN", "GREEN"],
            "weather_current": ["CLEAR", "CLEAR", "CLEAR", "CLEAR", "CLEAR", "CLEAR", "CLEAR", "CLEAR"],
            "circuit_length_km": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "circuit_turns": [18, 18, 18, 18, 18, 18, 18, 18],
            "circuit_drs_zones": [1, 1, 1, 1, 1, 1, 1, 1],
            "circuit_overtake_difficulty": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            "circuit_base_lap_time_sec": [88.0, 88.0, 88.0, 88.0, 88.0, 88.0, 88.0, 88.0],
            "tire_wear_pct": [0.2, 0.5, 0.8, 0.4, 0.9, 1.5, 0.6, 0.9],
            "pit_stop_this_lap": [0, 0, 0, 0, 0, 0, 0, 0],
            "pit_new_compound": ["HARD", "HARD", "HARD", "SOFT", "SOFT", "SOFT", "HARD", "HARD"],
        }
    )


def _test_rows():
    return pd.DataFrame(
        {
            "race_id": [3, 3, 3],
            "driver_id": [20, 20, 20],
            "season": [2024, 2024, 2024],
            "race_round": [3, 3, 3],
            "lap": [1, 2, 3],
            "tire_compound": ["MEDIUM", "MEDIUM", "MEDIUM"],
            "tire_age_laps": [1, 2, 3],
            "lap_time_sec": [95.0, 97.0, 98.0],
            "s1_time_sec": [30.0, 31.0, 32.0],
            "s2_time_sec": [31.0, 32.0, 33.0],
            "s3_time_sec": [34.0, 34.0, 33.0],
            "fuel_load_kg": [99.0, 92.0, 88.0],
            "ers_deploy_pct": [28.0, 30.0, 36.0],
            "ers_harvest_pct": [22.0, 24.0, 26.0],
            "driver_aggression_rating": [75.0, 77.0, 79.0],
            "driver_consistency_rating": [81.0, 82.0, 83.0],
            "driver_skill_rating": [81.0, 82.0, 83.0],
            "race_air_temp_c": [28.0, 29.0, 30.0],
            "race_track_temp_c": [35.0, 36.0, 37.0],
            "race_humidity_pct": [43.0, 44.0, 46.0],
            "wind_speed_kph": [5.0, 5.5, 6.0],
            "track_grip_level": [0.91, 0.89, 0.87],
            "gap_ahead_sec": [2.0, 1.0, 0.7],
            "gap_behind_sec": [1.5, 2.0, 1.0],
            "gap_to_leader_sec": [4.0, 3.0, 2.0],
            "position": [2, 2, 1],
            "grid_position": [2, 2, 2],
            "race_total_laps": [10, 10, 10],
            "track_status": ["GREEN", "GREEN", "GREEN"],
            "weather_current": ["CLEAR", "CLEAR", "CLEAR"],
            "circuit_length_km": [5.0, 5.0, 5.0],
            "circuit_turns": [18, 18, 18],
            "circuit_drs_zones": [1, 1, 1],
            "circuit_overtake_difficulty": [0.6, 0.6, 0.6],
            "circuit_base_lap_time_sec": [88.0, 88.0, 88.0],
            "tire_wear_pct": [0.7, 1.2, 1.8],
            "pit_stop_this_lap": [0, 0, 0],
            "pit_new_compound": ["MEDIUM", "MEDIUM", "MEDIUM"],
        }
    )


def test_original_derived_dataset_tracks_source_lineage_and_target():
    train = _train_rows()
    test = _test_rows()

    derived = build_original_derived_dataset(train, test, seed=42)

    assert {"_source_file", "_source_row_number", "next_lap_wear_increment"}.issubset(derived.columns)
    assert set(derived["_source_file"].unique()) <= {"train.csv", "test.csv"}
    assert "tire_wear_pct" in derived.columns
    assert "next_lap_degradation_pct" not in derived.columns
    assert "future_tire_wear_pct" not in derived.columns
    assert "next_lap_lap_time_sec" not in derived.columns
    assert derived["next_lap_wear_increment"].notna().all()
    assert len(derived) > 0


def test_original_derived_target_changes_with_current_state_and_excludes_legacy_target():
    train = _train_rows()
    test = _test_rows()

    derived = build_original_derived_dataset(train, test, seed=99)
    same_compound_age = derived[(derived["tire_compound"] == "HARD") & (derived["tire_age_laps"] == 2)].copy()
    assert same_compound_age["next_lap_wear_increment"].nunique() > 1

    assert not any(column.lower().startswith("future_") for column in derived.columns)
    assert not any(column.lower().startswith("next_") for column in derived.columns if column.lower() != "next_lap_wear_increment")
    assert "tire_wear_pct" in derived.columns


def test_split_benchmark_parts_are_deterministic_and_complete():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "derived"
    part_paths = sorted(data_dir.glob("original_derived_seed_42_part*.csv"))
    assert part_paths, "split benchmark files must exist"
    assert all(path.stat().st_size < 100 * 1024 * 1024 for path in part_paths), "each split CSV must remain under GitHub's per-file limit"

    frames = [pd.read_csv(path) for path in part_paths]
    combined = pd.concat(frames, ignore_index=True)
    expected = build_original_derived_dataset(pd.read_csv(data_dir.parent.parent / "train.csv"), pd.read_csv(data_dir.parent.parent / "test.csv"), seed=42)

    assert list(combined.columns) == list(expected.columns)
    assert len(combined) == len(expected)
    assert combined.shape[0] == sum(frame.shape[0] for frame in frames)
    assert combined["_source_file"].tolist() == expected["_source_file"].tolist()
    assert combined["_source_row_number"].tolist() == expected["_source_row_number"].tolist()

    for column in combined.columns:
        if pd.api.types.is_numeric_dtype(combined[column]):
            assert np.allclose(combined[column].to_numpy(), expected[column].to_numpy(), equal_nan=True), column
        else:
            left = combined[column].tolist()
            right = expected[column].tolist()
            assert len(left) == len(right)
            for left_value, right_value in zip(left, right):
                if pd.isna(left_value) and pd.isna(right_value):
                    continue
                assert left_value == right_value, (column, left_value, right_value)
    assert combined["next_lap_wear_increment"].notna().all()

    loaded = load_split_benchmark(data_dir, seed=42)
    for column in loaded.columns:
        if pd.api.types.is_numeric_dtype(loaded[column]):
            assert np.allclose(loaded[column].to_numpy(), expected[column].to_numpy(), equal_nan=True), column
        else:
            left = loaded[column].tolist()
            right = expected[column].tolist()
            assert len(left) == len(right)
            for left_value, right_value in zip(left, right):
                if pd.isna(left_value) and pd.isna(right_value):
                    continue
                assert left_value == right_value, (column, left_value, right_value)
