from __future__ import annotations

import json
from pathlib import Path

import joblib

from .config import CORRECTED_MODEL_ARTIFACT_DIR, DEFAULT_MODEL_NAMES, PROJECT_ROOT, TARGET_NAME

MODEL_REGISTRY_PATH = CORRECTED_MODEL_ARTIFACT_DIR / "registry.json"


def load_corrected_model_registry() -> dict:
    if MODEL_REGISTRY_PATH.exists():
        return json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "artifact_scope": "corrected_chronological_synthetic_benchmark",
        "models": {
            name: {"artifact_path": f"artifacts/models/corrected/{name}.joblib"}
            for name in DEFAULT_MODEL_NAMES
            if (CORRECTED_MODEL_ARTIFACT_DIR / f"{name}.joblib").exists()
        },
    }


def _registry_models() -> dict:
    registry = load_corrected_model_registry()
    models = registry.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("Corrected model registry is malformed: 'models' must be an object.")
    return models


def _corrected_model_path(model_name: str) -> Path:
    registry_entry = _registry_models().get(model_name, {})
    artifact_path = registry_entry.get("artifact_path") if isinstance(registry_entry, dict) else None
    if artifact_path:
        candidate = Path(artifact_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
    else:
        candidate = CORRECTED_MODEL_ARTIFACT_DIR / f"{model_name}.joblib"
    return candidate


def list_available_models() -> list[str]:
    return [
        name
        for name in DEFAULT_MODEL_NAMES
        if name in _registry_models() and _corrected_model_path(name).exists()
    ]


def get_model_metadata(model_name: str) -> dict:
    artifact = load_model(model_name)
    if isinstance(artifact, dict):
        return artifact.get("metadata", artifact)
    return {"model": model_name}


def load_model(model_name: str):
    cleaned = str(model_name).strip().lower()
    if cleaned not in DEFAULT_MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}. Available: {DEFAULT_MODEL_NAMES}")

    candidate = _corrected_model_path(cleaned)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Corrected model artifact not found for '{cleaned}' at {candidate}. "
            "Run the explicit corrected training/finalization command first."
        )
    artifact = joblib.load(candidate)
    _validate_corrected_artifact(cleaned, artifact, candidate)
    return artifact


def _validate_corrected_artifact(model_name: str, artifact, path: Path) -> None:
    if not isinstance(artifact, dict):
        raise TypeError(f"Corrected model artifact at {path} is not a dictionary payload.")
    if artifact.get("target") != TARGET_NAME:
        raise ValueError(f"Corrected model artifact at {path} has unexpected target {artifact.get('target')!r}.")
    if artifact.get("model") is None:
        raise KeyError(f"Corrected model artifact at {path} does not include a trained estimator.")
    feature_columns = artifact.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"Corrected model artifact at {path} does not include feature columns.")
    metadata = artifact.get("metadata", {})
    if metadata.get("model_name") != model_name:
        raise ValueError(f"Corrected model artifact at {path} has inconsistent model metadata.")
    if metadata.get("preprocessing_fit_scope") != "train only":
        raise ValueError(f"Corrected model artifact at {path} does not declare train-only preprocessing.")
    if metadata.get("dataset_sha256") != load_corrected_model_registry().get("dataset", {}).get("sha256"):
        raise ValueError(f"Corrected model artifact at {path} does not match the corrected registry dataset hash.")
    if metadata.get("feature_columns") != feature_columns:
        raise ValueError(f"Corrected model artifact at {path} has inconsistent feature-column metadata.")


def save_model_artifact(model_name: str, payload: dict, directory: str | Path | None = None):
    target = Path(directory) if directory is not None else CORRECTED_MODEL_ARTIFACT_DIR
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
