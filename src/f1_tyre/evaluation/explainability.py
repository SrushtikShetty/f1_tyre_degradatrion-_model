from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def grouped_permutation_importance(
    model: Any,
    validation_features: pd.DataFrame,
    validation_target: Sequence[float],
    feature_groups: Mapping[str, Sequence[str]],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure validation MAE increase when feature groups are permuted together."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    target = np.asarray(validation_target, dtype=float)
    if len(target) != len(validation_features) or len(target) < 2:
        raise ValueError("Validation features and target must have matching lengths of at least two.")

    baseline_prediction = np.asarray(model.predict(validation_features), dtype=float)
    baseline_mae = float(mean_absolute_error(target, baseline_prediction))
    rng = np.random.default_rng(random_state)
    results = []
    for group_name, columns in feature_groups.items():
        group_columns = tuple(dict.fromkeys(str(column) for column in columns))
        if not group_columns:
            raise ValueError(f"Feature group {group_name!r} cannot be empty.")
        missing = set(group_columns).difference(validation_features.columns)
        if missing:
            raise ValueError(f"Feature group {group_name!r} has unknown columns: {sorted(missing)}")

        increases = []
        for _ in range(n_repeats):
            permutation = rng.permutation(len(validation_features))
            permuted = validation_features.copy()
            permuted.loc[:, list(group_columns)] = validation_features.iloc[permutation][list(group_columns)].to_numpy()
            prediction = np.asarray(model.predict(permuted), dtype=float)
            increases.append(float(mean_absolute_error(target, prediction)) - baseline_mae)

        results.append(
            {
                "feature_group": str(group_name),
                "features": list(group_columns),
                "importance_mean": float(np.mean(increases)),
                "importance_std": float(np.std(increases, ddof=1)) if len(increases) > 1 else 0.0,
                "baseline_mae": baseline_mae,
                "method": "grouped_permutation_validation",
                "metric": "increase_in_mae",
            }
        )
    return pd.DataFrame(results).sort_values("importance_mean", ascending=False).reset_index(drop=True)