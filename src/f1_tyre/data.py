from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMON_GROUP_COLS, LEGACY_TARGET_NAME, MAIN_TARGET, TARGET_NAME

SIMULATED_TARGET_COLUMN = "next_lap_degradation_pct"
TARGET_ONLY_COLUMNS = {
    MAIN_TARGET,
    LEGACY_TARGET_NAME,
    TARGET_NAME,
    SIMULATED_TARGET_COLUMN,
}


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_dataset(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir) if data_dir is not None else Path.cwd()
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find dataset: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Could not find dataset: {test_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def build_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build rows labeled by the simulator's explicit next-lap outcome.

    The target is generated from end-of-lap state at t and represents wear
    realized during t+1. This function does not shift or inspect future rows.
    """
    require_columns(df, ["race_id", "driver_id", "lap", SIMULATED_TARGET_COLUMN])
    out = df.copy().sort_values(COMMON_GROUP_COLS + ["lap"]).reset_index(drop=True)
    out[TARGET_NAME] = pd.to_numeric(out[SIMULATED_TARGET_COLUMN], errors="coerce")
    out = out.loc[out[TARGET_NAME].notna()].copy()
    target_proxies = [
        column
        for column in out.columns
        if column in TARGET_ONLY_COLUMNS - {TARGET_NAME}
        or (column != TARGET_NAME and str(column).lower().startswith(("next_", "future_")))
    ]
    return out.drop(columns=target_proxies).reset_index(drop=True)


def build_legacy_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the historical shift-derived target for audit reproduction only."""
    require_columns(df, ["race_id", "driver_id", "lap", "tire_age_laps", "tire_compound", MAIN_TARGET])
    out = df.copy().sort_values(COMMON_GROUP_COLS + ["lap"]).reset_index(drop=True)
    grouped = out.groupby(COMMON_GROUP_COLS, sort=False)
    out["_next_lap"] = grouped["lap"].shift(-1)
    out["_next_tire_age"] = grouped["tire_age_laps"].shift(-1)
    out["_next_compound"] = grouped["tire_compound"].shift(-1)
    out["_next_wear"] = grouped[MAIN_TARGET].shift(-1)
    out[TARGET_NAME] = out["_next_wear"] - out[MAIN_TARGET]
    out[LEGACY_TARGET_NAME] = out["_next_wear"]

    valid = (
        out["_next_lap"].eq(out["lap"] + 1)
        & out["_next_tire_age"].eq(out["tire_age_laps"] + 1)
        & out["_next_compound"].astype("string").eq(out["tire_compound"].astype("string"))
        & out["_next_wear"].notna()
    )
    return out.loc[valid].drop(
        columns=["_next_lap", "_next_tire_age", "_next_compound", "_next_wear"]
    ).reset_index(drop=True)


def build_model_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Select and audit X, refusing target or future-state columns explicitly."""
    requested = [str(column) for column in feature_columns]
    forbidden = [
        column
        for column in requested
        if column in TARGET_ONLY_COLUMNS
        or column.lower().startswith(("next_", "future_"))
    ]
    if forbidden:
        raise ValueError(f"Target or future-state columns cannot enter X: {sorted(set(forbidden))}")
    require_columns(df, requested)
    features = df[requested].copy()
    from .evaluation.leakage_audit import audit_feature_matrix

    audit_feature_matrix(features)
    return features


def ensure_causal_sort(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in COMMON_GROUP_COLS + ["lap"] if column in df.columns]
    return df.sort_values(sort_columns).reset_index(drop=True)


def drop_duplicate_race_laps(df: pd.DataFrame) -> pd.DataFrame:
    key = [column for column in COMMON_GROUP_COLS + ["lap"] if column in df.columns]
    if not key:
        return df.copy()
    deduped = df.drop_duplicates(subset=key, keep="first")
    return deduped.sort_values(key).reset_index(drop=True)


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out
