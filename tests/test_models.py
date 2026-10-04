import json
from pathlib import Path

import pytest

from f1_tyre.config import TARGET_NAME
from f1_tyre.model_loader import list_available_models, load_corrected_model_registry, load_model
from models.compare_models import build_grouped_cv_report
from models.common import label_supplementary_grouped_cv


def test_model_registry_lists_supported_models():
    models = list_available_models()
    assert "ridge" in models
    assert "random_forest" in models
    assert "xgboost" in models
    assert "neural_network" in models


def test_model_loader_uses_corrected_artifact_with_authoritative_provenance():
    artifact = load_model("xgboost")
    registry = load_corrected_model_registry()

    assert artifact["target"] == TARGET_NAME
    assert len(artifact["feature_columns"]) == 50
    assert artifact["feature_columns"] == artifact["metadata"]["feature_columns"]
    assert artifact["metadata"]["dataset_sha256"] == registry["dataset"]["sha256"]
    assert artifact["metadata"]["preprocessing_fit_scope"] == "train only"
    assert artifact["metrics"]["final_holdout"]["r2"] == pytest.approx(0.9119949374093644)


def test_corrected_model_registry_matches_saved_artifacts():
    registry = load_corrected_model_registry()

    assert registry["artifact_scope"] == "corrected_chronological_synthetic_benchmark"
    assert set(registry["models"]) == {"ridge", "random_forest", "xgboost", "neural_network"}
    for model_name, entry in registry["models"].items():
        artifact = load_model(model_name)
        assert Path(entry["artifact_path"]).as_posix().endswith(f"{model_name}.joblib")
        assert entry["target"] == TARGET_NAME
        assert entry["feature_columns"] == artifact["feature_columns"]
        assert entry["metrics"] == json.loads(json.dumps(artifact["metrics"]))


def test_model_loader_does_not_fall_back_to_legacy_artifacts(monkeypatch, tmp_path):
    registry_path = tmp_path / "registry.json"
    missing_artifact = tmp_path / "missing_xgboost.joblib"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_scope": "corrected_chronological_synthetic_benchmark",
                "dataset": {"sha256": "not-used"},
                "models": {
                    "xgboost": {
                        "artifact_path": str(missing_artifact),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("f1_tyre.model_loader.MODEL_REGISTRY_PATH", registry_path)

    with pytest.raises(FileNotFoundError, match="Corrected model artifact not found"):
        load_model("xgboost")


def test_grouped_cv_metrics_are_machine_labeled_supplementary():
    metrics = label_supplementary_grouped_cv({"model": "Ridge", "r2": 0.8})

    assert metrics["metric_scope"] == "supplementary_grouped_cv"
    assert "GroupKFold" in metrics["metric_protocol"]
    assert metrics["primary_metric"] is False


def test_grouped_cv_report_declares_chronological_metrics_authoritative():
    report = build_grouped_cv_report([{"model": "Ridge", "r2": 0.8}])

    assert report["status"] == "SUPPLEMENTARY_GROUPED_CV_ONLY"
    assert report["metric_scope"] == "supplementary_grouped_cv"
    assert report["primary_metric_source"] == "artifacts/evaluation/final_metrics.json"
    assert report["models"][0]["model"] == "Ridge"
