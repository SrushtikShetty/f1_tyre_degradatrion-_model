import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_METRICS_PATH = PROJECT_ROOT / "artifacts" / "evaluation" / "final_metrics.json"
FINAL_REPORT_PATH = PROJECT_ROOT / "artifacts" / "evaluation" / "final_report.json"
AUTHORITATIVE_DATASET_PATH = PROJECT_ROOT / "data" / "generated" / "authoritative_seed_20261004.csv"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_authoritative_dataset_hash_and_row_counts_match_final_metrics():
    metrics = _json(FINAL_METRICS_PATH)
    dataset = pd.read_csv(AUTHORITATIVE_DATASET_PATH)
    labeled_rows = len(dataset.dropna(subset=["next_lap_degradation_pct"]))

    assert hashlib.sha256(AUTHORITATIVE_DATASET_PATH.read_bytes()).hexdigest() == metrics["dataset"]["sha256"]
    assert len(dataset) == metrics["dataset"]["raw_rows"]
    assert labeled_rows == metrics["dataset"]["modeling_rows"]
    assert metrics["dataset"]["rows"] == metrics["dataset"]["modeling_rows"]
    assert dataset["race_id"].nunique() == metrics["dataset"]["races"]


def test_final_report_references_same_authoritative_dataset_and_metrics():
    metrics = _json(FINAL_METRICS_PATH)
    report = _json(FINAL_REPORT_PATH)

    assert report["dataset_type"] == "synthetic"
    assert report["dataset"]["sha256"] == metrics["dataset"]["sha256"]
    assert report["dataset"]["raw_rows"] == metrics["dataset"]["raw_rows"]
    assert report["dataset"]["modeling_rows"] == metrics["dataset"]["modeling_rows"]
    assert report["metrics"] == metrics["metrics"]
    assert report["shap_computed"] is False


def test_final_metrics_protocol_and_explainability_are_authoritative():
    metrics = _json(FINAL_METRICS_PATH)

    assert metrics["evaluation_protocol"]["integrity"] == {
        "zero_race_overlap": True,
        "zero_race_driver_overlap": True,
        "zero_tyre_stint_overlap": True,
        "zero_duplicate_row_overlap": True,
    }
    assert metrics["preprocessing_fit_scope"] == "train only"
    assert metrics["holdout_evaluated_once"] is True
    assert metrics["target_determinism_guard"]["passed"] is True
    assert metrics["target_determinism_guard"]["final_holdout_compound_age_lookup_r2"] < 0.98
    assert metrics["explainability"]["method"] == "grouped_permutation_validation"
    assert metrics["explainability"]["metric"] == "increase_in_mae"
    assert metrics["explainability"]["shap_computed"] is False
