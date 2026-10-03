from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


PARTITION_NAMES = ("train", "validation", "holdout")
ROW_KEY_COLUMNS = ("race_id", "driver_id", "lap")
RACE_DRIVER_COLUMNS = ("race_id", "driver_id")


def _chronological_race_order(df: pd.DataFrame) -> list[Any]:
    race_column = "race_id"
    if race_column not in df:
        raise ValueError("Chronological splitting requires race_id.")

    has_event_date = "race_race_date" in df
    has_season_round = {"season", "race_round"}.issubset(df.columns)
    if has_event_date or has_season_round:
        metadata_columns = [
            column
            for column in ("season", "race_round", "race_race_date")
            if column in df
        ]
        consistency = df.groupby(race_column)[metadata_columns].nunique(dropna=False)
        if consistency.gt(1).any().any():
            raise ValueError("Chronology metadata must be constant within each race.")
        metadata = df[[race_column, *metadata_columns]].drop_duplicates(subset=[race_column]).copy()
        if "race_race_date" in metadata:
            metadata["race_race_date"] = pd.to_datetime(metadata["race_race_date"], errors="raise")
        order_columns = [column for column in ("season", "race_race_date", "race_round", race_column) if column in metadata]
        return metadata.sort_values(order_columns, kind="mergesort")[race_column].tolist()

    numeric_race_ids = pd.to_numeric(df[race_column], errors="coerce")
    if numeric_race_ids.isna().any():
        raise ValueError("Without date/season-round metadata, race_id must be numeric and chronological.")
    metadata = pd.DataFrame({race_column: df[race_column], "_race_order": numeric_race_ids}).drop_duplicates(race_column)
    return metadata.sort_values(["_race_order", race_column], kind="mergesort")[race_column].tolist()


def _key_set(df: pd.DataFrame, columns: tuple[str, ...]) -> set[tuple[Any, ...]]:
    return set(df[list(columns)].itertuples(index=False, name=None))


def _stint_key_set(df: pd.DataFrame) -> set[tuple[Any, ...]]:
    explicit_column = next((name for name in ("tire_stint_id", "tyre_stint_id", "stint_id") if name in df), None)
    if explicit_column:
        return _key_set(df, ("race_id", "driver_id", explicit_column))
    if not {"tire_age_laps", "tire_compound"}.issubset(df.columns):
        return _key_set(df, RACE_DRIVER_COLUMNS)

    ordered = df.sort_values([*RACE_DRIVER_COLUMNS, "lap"], kind="mergesort").reset_index(drop=True)
    grouped = ordered.groupby(list(RACE_DRIVER_COLUMNS), sort=False)
    previous_age = grouped["tire_age_laps"].shift(1)
    previous_compound = grouped["tire_compound"].shift(1)
    starts_stint = (
        previous_age.isna()
        | ordered["tire_age_laps"].le(previous_age)
        | ordered["tire_compound"].ne(previous_compound)
    )
    stint_number = starts_stint.astype(int).groupby(
        [ordered[column] for column in RACE_DRIVER_COLUMNS], sort=False
    ).cumsum()
    return set(zip(ordered["race_id"], ordered["driver_id"], stint_number))


def validate_split_integrity(partitions: Mapping[str, pd.DataFrame]) -> dict[str, bool]:
    if set(partitions) != set(PARTITION_NAMES):
        raise ValueError(f"Partitions must be exactly {PARTITION_NAMES}.")
    for name, frame in partitions.items():
        missing = set(ROW_KEY_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required row-key columns: {sorted(missing)}")
        if frame[list(ROW_KEY_COLUMNS)].isna().any().any():
            raise ValueError(f"{name} contains null race-driver-lap keys.")
        if frame.duplicated(list(ROW_KEY_COLUMNS)).any():
            raise ValueError(f"{name} contains duplicate race-driver-lap rows.")

    for index, left_name in enumerate(PARTITION_NAMES):
        for right_name in PARTITION_NAMES[index + 1 :]:
            left = partitions[left_name]
            right = partitions[right_name]
            for label, columns in (
                ("race", ("race_id",)),
                ("race-driver", RACE_DRIVER_COLUMNS),
            ):
                if _key_set(left, columns) & _key_set(right, columns):
                    raise ValueError(f"{label} overlap between {left_name} and {right_name}.")
            if _stint_key_set(left) & _stint_key_set(right):
                raise ValueError(f"Tyre-stint overlap between {left_name} and {right_name}.")

    return {
        "zero_race_overlap": True,
        "zero_race_driver_overlap": True,
        "zero_tyre_stint_overlap": True,
        "zero_duplicate_row_overlap": True,
    }


def chronological_race_splits(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Split whole races chronologically; the final partition is holdout-only."""
    missing = set(ROW_KEY_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Chronological splitting requires columns: {sorted(missing)}")
    if len(fractions) != 3 or any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be three positive values that sum to 1.")
    if df[list(ROW_KEY_COLUMNS)].isna().any().any():
        raise ValueError("Chronological splitting does not accept null race-driver-lap keys.")
    if df.duplicated(list(ROW_KEY_COLUMNS)).any():
        raise ValueError("Chronological splitting does not accept duplicate race-driver-lap rows.")

    race_order = _chronological_race_order(df)
    if len(race_order) < 6:
        raise ValueError("At least six races are required for non-empty train/validation/holdout splits.")
    train_end = int(len(race_order) * fractions[0])
    validation_end = int(len(race_order) * (fractions[0] + fractions[1]))
    race_ids = {
        "train": race_order[:train_end],
        "validation": race_order[train_end:validation_end],
        "holdout": race_order[validation_end:],
    }
    if any(not values for values in race_ids.values()):
        raise ValueError("Chronological split produced an empty partition.")

    partitions = {
        name: df.loc[df["race_id"].isin(ids)].copy().reset_index(drop=True)
        for name, ids in race_ids.items()
    }
    integrity = validate_split_integrity(partitions)
    metadata: dict[str, Any] = {
        "protocol": "chronological_race_disjoint",
        "fractions": dict(zip(PARTITION_NAMES, fractions)),
        "race_ids": race_ids,
        "race_counts": {name: len(ids) for name, ids in race_ids.items()},
        "row_counts": {name: len(frame) for name, frame in partitions.items()},
        "integrity": integrity,
        "holdout_role": "final evaluation only after model and feature selection are complete",
    }
    return partitions, metadata