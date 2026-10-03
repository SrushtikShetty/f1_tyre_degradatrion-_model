from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .baselines import fit_baseline_models
from ..strict_feature_policy import ALLOWED_FEATURES, PARTITION_ONLY_FEATURES


DEFAULT_TARGET = "next_lap_wear_increment"
DEFAULT_GROUP_COLUMNS = ("tire_compound", "tire_age_laps")


def _variance_explained_by_groups(frame: pd.DataFrame, target_column: str, group_columns: tuple[str, ...]) -> float:
    actual = frame[target_column].to_numpy(dtype=float)
    total_sum_squares = float(np.square(actual - actual.mean()).sum())
    if total_sum_squares == 0.0:
        return 0.0
    group_means = frame.groupby(list(group_columns), dropna=False)[target_column].transform("mean").to_numpy(dtype=float)
    residual_sum_squares = float(np.square(actual - group_means).sum())
    return float(1.0 - residual_sum_squares / total_sum_squares)


def _conditional_binned_entropy(
    frame: pd.DataFrame,
    target_column: str,
    group_columns: tuple[str, ...],
    bin_width: float,
) -> float:
    if bin_width <= 0:
        raise ValueError("entropy_bin_width must be positive.")
    grouped = frame[list(group_columns)].copy()
    grouped["_target_bin"] = np.floor(frame[target_column].to_numpy(dtype=float) / bin_width).astype(int)
    total = len(grouped)
    entropy = 0.0
    for _, group in grouped.groupby(list(group_columns), dropna=False, sort=False):
        probabilities = group["_target_bin"].value_counts(normalize=True).to_numpy(dtype=float)
        entropy_by_group = float(-np.sum(probabilities * np.log2(probabilities)))
        entropy += len(group) / total * entropy_by_group
    return float(entropy)


def compute_target_structure(
    frame: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    group_columns: tuple[str, ...] = DEFAULT_GROUP_COLUMNS,
    entropy_bin_width: float = 0.05,
) -> dict[str, Any]:
    missing = {target_column, *group_columns}.difference(frame.columns)
    if missing:
        raise ValueError(f"Target diagnostic is missing columns: {sorted(missing)}")
    clean = frame.loc[frame[target_column].notna()].copy()
    target = pd.to_numeric(clean[target_column], errors="raise").astype(float)
    if clean.empty or not np.isfinite(target.to_numpy()).all():
        raise ValueError("Target diagnostics require finite target values.")
    clean[target_column] = target
    grouped = clean.groupby(list(group_columns), dropna=False)[target_column]
    unique_per_group = grouped.nunique()
    std_per_group = grouped.std().fillna(0.0)
    numeric_correlations = {}
    for column in clean.columns:
        if column == target_column or column not in ALLOWED_FEATURES or column in PARTITION_ONLY_FEATURES:
            continue
        if not pd.api.types.is_numeric_dtype(clean[column]) or clean[column].nunique(dropna=True) < 2:
            continue
        paired = pd.concat([clean[column], clean[target_column]], axis=1).dropna()
        if len(paired) >= 2:
            numeric_correlations[column] = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))

    return {
        "row_count": int(len(clean)),
        "target_unique_values": int(target.nunique()),
        "target_mean": float(target.mean()),
        "target_variance": float(target.var(ddof=1)) if len(target) > 1 else 0.0,
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "compound_age_group_count": int(grouped.ngroups),
        "groups_with_multiple_targets": int((unique_per_group > 1).sum()),
        "groups_with_zero_sample_std": int((std_per_group == 0).sum()),
        "group_unique_target_count": {
            "min": int(unique_per_group.min()),
            "mean": float(unique_per_group.mean()),
            "max": int(unique_per_group.max()),
        },
        "within_group_target_std": {
            "mean": float(std_per_group.mean()),
            "median": float(std_per_group.median()),
            "max": float(std_per_group.max()),
        },
        "conditional_binned_entropy_bits": _conditional_binned_entropy(
            clean, target_column, group_columns, entropy_bin_width
        ),
        "entropy_bin_width": float(entropy_bin_width),
        "variance_explained_by_compound": _variance_explained_by_groups(
            clean, target_column, ("tire_compound",)
        ),
        "variance_explained_by_age": _variance_explained_by_groups(
            clean, target_column, ("tire_age_laps",)
        ),
        "variance_explained_by_compound_and_age": _variance_explained_by_groups(
            clean, target_column, group_columns
        ),
        "numeric_feature_correlations": numeric_correlations,
    }


def diagnose_target_determinism(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    evaluation_split: str = "validation",
    threshold: float = 0.98,
    deterministic_demo: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if evaluation_split == "holdout":
        raise ValueError("Do not inspect the final holdout with this development diagnostic.")

    structure = compute_target_structure(train, target_column)
    baseline = fit_baseline_models(train, target_column)["CompoundAgeLookup"]
    valid = evaluation.loc[evaluation[target_column].notna()].copy()
    actual = pd.to_numeric(valid[target_column], errors="raise").to_numpy(dtype=float)
    if len(actual) < 2:
        raise ValueError("At least two evaluation targets are required for R2.")
    predicted, unseen_keys = baseline.predict(valid)
    lookup_r2 = float(r2_score(actual, predicted))
    report = {
        "status": "DETERMINISTIC_DEMO_EXCEPTION" if deterministic_demo and lookup_r2 >= threshold else "PASS",
        "evaluation_split": evaluation_split,
        "threshold": float(threshold),
        "deterministic_demo": bool(deterministic_demo),
        "compound_age_lookup": {
            "r2": lookup_r2,
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "rows": int(len(actual)),
            "unseen_keys": int(unseen_keys),
        },
        "training_target_structure": structure,
    }
    if lookup_r2 >= threshold and not deterministic_demo:
        raise ValueError(
            f"Compound-age lookup R2 {lookup_r2:.6f} meets/exceeds the {threshold:.2f} determinism guard."
        )
    return report