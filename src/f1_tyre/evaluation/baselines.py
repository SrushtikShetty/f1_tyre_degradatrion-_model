from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class MedianLookupBaseline:
    name: str
    key_columns: tuple[str, ...]
    lookup: dict[tuple[Any, ...], float]
    fallback_value: float

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, int]:
        if not self.key_columns:
            return np.full(len(frame), self.fallback_value, dtype=float), 0
        missing = set(self.key_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Prediction frame is missing baseline keys: {sorted(missing)}")

        predictions = []
        unseen_keys = 0
        for key in frame[list(self.key_columns)].itertuples(index=False, name=None):
            if key not in self.lookup:
                unseen_keys += 1
                predictions.append(self.fallback_value)
            else:
                predictions.append(self.lookup[key])
        return np.asarray(predictions, dtype=float), unseen_keys


def _lookup_from_training(
    train: pd.DataFrame,
    target_column: str,
    key_columns: tuple[str, ...],
) -> dict[tuple[Any, ...], float]:
    grouped = train.groupby(list(key_columns), dropna=False, sort=False)[target_column].median()
    if len(key_columns) == 1:
        return {(key,): float(value) for key, value in grouped.items()}
    return {tuple(key): float(value) for key, value in grouped.items()}


def fit_baseline_models(
    train: pd.DataFrame,
    target_column: str = "next_lap_wear_increment",
) -> dict[str, MedianLookupBaseline]:
    """Fit all baseline statistics using TRAIN only."""
    required = {target_column, "tire_compound", "tire_age_laps"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training frame is missing baseline columns: {sorted(missing)}")
    clean_train = train.loc[train[target_column].notna()].copy()
    if clean_train.empty:
        raise ValueError("Cannot fit baselines on an empty training partition.")

    target = pd.to_numeric(clean_train[target_column], errors="raise").astype(float)
    global_mean = float(target.mean())
    global_median = float(target.median())
    compound_lookup = _lookup_from_training(clean_train, target_column, ("tire_compound",))
    compound_age_lookup = _lookup_from_training(
        clean_train,
        target_column,
        ("tire_compound", "tire_age_laps"),
    )
    return {
        "GlobalMean": MedianLookupBaseline("GlobalMean", (), {(): global_mean}, global_mean),
        "CompoundOnly": MedianLookupBaseline("CompoundOnly", ("tire_compound",), compound_lookup, global_median),
        "CompoundAgeLookup": MedianLookupBaseline(
            "CompoundAgeLookup",
            ("tire_compound", "tire_age_laps"),
            compound_age_lookup,
            global_median,
        ),
    }


def score_baselines(
    baselines: dict[str, MedianLookupBaseline],
    validation: pd.DataFrame,
    target_column: str = "next_lap_wear_increment",
    evaluation_split: str = "validation",
    allow_final_holdout: bool = False,
) -> list[dict[str, Any]]:
    """Score TRAIN-only baselines; holdout scoring requires explicit finalization."""
    if evaluation_split not in {"validation", "final_holdout"}:
        raise ValueError("evaluation_split must be 'validation' or 'final_holdout'.")
    if evaluation_split == "final_holdout" and not allow_final_holdout:
        raise ValueError("Final holdout scoring requires explicit final evaluation authorization.")
    if target_column not in validation:
        raise ValueError(f"Validation frame is missing target column {target_column!r}.")
    valid = validation.loc[validation[target_column].notna()].copy()
    actual = pd.to_numeric(valid[target_column], errors="raise").to_numpy(dtype=float)
    if len(actual) < 2:
        raise ValueError("At least two validation targets are required for R2.")

    results = []
    for name, baseline in baselines.items():
        predicted, unseen_keys = baseline.predict(valid)
        results.append(
            {
                "model": name,
                "split": evaluation_split,
                "metric_scope": "final_holdout_baseline" if evaluation_split == "final_holdout" else "baseline_validation",
                "r2": float(r2_score(actual, predicted)),
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "number_of_rows": int(len(actual)),
                "unseen_lookup_keys": int(unseen_keys),
            }
        )
    return results