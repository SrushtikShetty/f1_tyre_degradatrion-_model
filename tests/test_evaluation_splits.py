import pandas as pd
import pytest

from f1_tyre.evaluation.splits import chronological_race_splits, validate_split_integrity


def _race_rows(race_count=10):
    rows = []
    for race_id in range(1, race_count + 1):
        for driver_id in (1, 2):
            for lap in (1, 2, 3):
                rows.append(
                    {
                        "race_id": race_id,
                        "season": 2025 + (race_id - 1) // 6,
                        "race_round": (race_id - 1) % 6 + 1,
                        "driver_id": driver_id,
                        "lap": lap,
                        "tire_compound": "MEDIUM",
                        "tire_age_laps": lap,
                    }
                )
    return pd.DataFrame(rows)


def test_splits_are_chronological_and_race_driver_stint_disjoint():
    partitions, metadata = chronological_race_splits(_race_rows())

    assert metadata["race_counts"] == {"train": 6, "validation": 2, "holdout": 2}
    assert metadata["race_ids"]["train"][-1] < metadata["race_ids"]["validation"][0]
    assert metadata["race_ids"]["validation"][-1] < metadata["race_ids"]["holdout"][0]
    assert metadata["integrity"] == {
        "zero_race_overlap": True,
        "zero_race_driver_overlap": True,
        "zero_tyre_stint_overlap": True,
        "zero_duplicate_row_overlap": True,
    }
    assert set(partitions) == {"train", "validation", "holdout"}


def test_duplicate_race_driver_lap_rows_are_rejected():
    rows = _race_rows()
    rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        chronological_race_splits(rows)


def test_inconsistent_chronology_within_a_race_is_rejected():
    rows = _race_rows()
    rows.loc[1, "season"] += 1

    with pytest.raises(ValueError, match="constant within each race"):
        chronological_race_splits(rows)


def test_split_integrity_rejects_shared_race_driver_even_with_different_laps():
    rows = _race_rows()
    partitions = {
        "train": rows.loc[(rows.race_id == 1) & (rows.lap == 1)],
        "validation": rows.loc[(rows.race_id == 1) & (rows.lap == 2)],
        "holdout": rows.loc[rows.race_id == 2],
    }

    with pytest.raises(ValueError, match="race overlap"):
        validate_split_integrity(partitions)


def test_non_numeric_race_ids_require_explicit_chronology():
    rows = _race_rows()
    rows["race_id"] = rows["race_id"].map(lambda value: f"race-{value}")
    rows = rows.drop(columns=["season", "race_round"])

    with pytest.raises(ValueError, match="must be numeric and chronological"):
        chronological_race_splits(rows)


def test_partial_season_metadata_does_not_establish_chronology():
    rows = _race_rows().drop(columns="race_round")
    rows["race_id"] = rows["race_id"].map(lambda value: f"race-{value}")

    with pytest.raises(ValueError, match="must be numeric and chronological"):
        chronological_race_splits(rows)