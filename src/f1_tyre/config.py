from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACT_DIR = ARTIFACTS_DIR / "models"
CORRECTED_MODEL_ARTIFACT_DIR = MODEL_ARTIFACT_DIR / "corrected"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
FEATURE_IMPORTANCE_DIR = ARTIFACTS_DIR / "feature_importance"

DEFAULT_MODEL_NAMES = ["ridge", "random_forest", "xgboost", "neural_network"]
COMMON_GROUP_COLS = ["race_id", "driver_id"]
TARGET_NAME = "next_lap_wear_increment"
LEGACY_TARGET_NAME = "future_tire_wear_pct"
MAIN_TARGET = "tire_wear_pct"

FORBIDDEN_LEAKAGE_PATTERNS = (
    "target_",
    "future_",
    "next_",
    "finish",
    "points",
    "fastest",
    "pit_stop",
    "pit_new",
    "race_end",
    "race_result",
)
