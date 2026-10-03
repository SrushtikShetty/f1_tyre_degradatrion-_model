from __future__ import annotations

import json
from pathlib import Path

import joblib

from .config import ARTIFACTS_DIR, DEFAULT_MODEL_NAMES, MODEL_ARTIFACT_DIR, PROJECT_ROOT

LEGACY_MODEL_PATH = PROJECT_ROOT / "f1_tyre_wear_model.joblib"


def _candidate_dirs() -> list[Path]:
    dirs = [MODEL_ARTIFACT_DIR, ARTIFACTS_DIR, PROJECT_ROOT]
    seen = set()
    ordered = []
    for path in dirs:
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def list_available_models() -> list[str]:
    names = []
    for directory in _candidate_dirs():
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix == ".joblib":
                stem = path.stem.lower()
                if stem in DEFAULT_MODEL_NAMES:
                    names.append(stem)
                if path.name.lower() == LEGACY_MODEL_PATH.name.lower():
                    names.extend(DEFAULT_MODEL_NAMES)
    for name in DEFAULT_MODEL_NAMES:
        if name not in names:
            names.append(name)
    return list(dict.fromkeys(names))


def get_model_metadata(model_name: str) -> dict:
    artifact = load_model(model_name)
    if isinstance(artifact, dict):
        return artifact.get("metadata", artifact)
    return {"model": model_name}


def load_model(model_name: str):
    cleaned = str(model_name).strip().lower()
    if cleaned not in DEFAULT_MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}. Available: {DEFAULT_MODEL_NAMES}")

    for directory in _candidate_dirs():
        candidate = directory / f"{cleaned}.joblib"
        if candidate.exists():
            return joblib.load(candidate)
    if LEGACY_MODEL_PATH.exists():
        return joblib.load(LEGACY_MODEL_PATH)
    raise FileNotFoundError(f"Trained model artifact not found for '{model_name}'. Run the explicit training command first.")


def save_model_artifact(model_name: str, payload: dict, directory: str | Path | None = None):
    target = Path(directory) if directory is not None else MODEL_ARTIFACT_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{model_name}.joblib"
    joblib.dump(payload, path)
    return path


def load_model_metrics(model_name: str) -> dict:
    artifact = load_model(model_name)
    if isinstance(artifact, dict):
        metrics = artifact.get("metrics") or artifact.get("oof_metrics") or {}
        return metrics
    return {}
