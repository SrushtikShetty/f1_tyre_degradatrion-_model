from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import COMMON_GROUP_COLS, MAIN_TARGET, TARGET_NAME


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
    require_columns(df, ["race_id", "driver_id", "lap", "tire_age_laps", "tire_compound", MAIN_TARGET])
    out = df.copy().sort_values(COMMON_GROUP_COLS + ["lap"]).reset_index(drop=True)
    grouped = out.groupby(COMMON_GROUP_COLS, sort=False)
    out["_next_lap"] = grouped["lap"].shift(-1)
    out["_next_compound"] = grouped["tire_compound"].shift(-1)
    out[TARGET_NAME] = grouped[MAIN_TARGET].shift(-1)

    valid = (
        out["_next_lap"].eq(out["lap"] + 1)
        & out["_next_compound"].astype("string").eq(out["tire_compound"].astype("string"))
        & out[TARGET_NAME].notna()
    )

    prepared = out.loc[valid].drop(columns=["_next_lap", "_next_compound"]).copy()
    prepared = prepared.reset_index(drop=True)
    return prepared


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
