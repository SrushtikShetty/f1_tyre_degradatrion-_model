from f1_tyre.model_loader import list_available_models
from models.compare_models import build_grouped_cv_report
from models.common import label_supplementary_grouped_cv


def test_model_registry_lists_supported_models():
    models = list_available_models()
    assert "ridge" in models
    assert "random_forest" in models
    assert "xgboost" in models
    assert "neural_network" in models


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
