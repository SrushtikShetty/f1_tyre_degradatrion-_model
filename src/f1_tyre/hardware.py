from __future__ import annotations

import warnings
from multiprocessing import cpu_count

from xgboost import XGBRegressor


def detect_xgboost_device() -> str:
    """Return 'cuda' when available, otherwise 'cpu'."""
    try:
        XGBRegressor(
            n_estimators=2,
            max_depth=2,
            learning_rate=0.1,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            random_state=42,
        ).fit([[0.0], [1.0]], [0.0, 1.0])
        return "cuda"
    except Exception:
        warnings.warn("CUDA unavailable for XGBoost; falling back to CPU.", RuntimeWarning)
        return "cpu"


def get_xgboost_settings(device: str | None = None) -> dict:
    chosen = device or detect_xgboost_device()
    return {
        "device": chosen,
        "n_jobs": max(1, cpu_count() - 2),
        "tree_method": "hist",
    }
