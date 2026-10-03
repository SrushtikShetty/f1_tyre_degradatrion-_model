import pandas as pd
import pytest

from f1_tyre.evaluation.target_diagnostics import compute_target_structure, diagnose_target_determinism


def _variable_frame():
    rows = []
    for compound, base in (("HARD", 0.5), ("SOFT", 0.9)):
        for age in (1, 2):
            for context in (0.0, 0.1, 0.2):
                rows.append(
                    {
                        "tire_compound": compound,
                        "tire_age_laps": age,
                        "race_track_temp_c": 30.0 + 20.0 * context,
                        "next_lap_wear_increment": base + age * 0.2 + context,
                    }
                )
    return pd.DataFrame(rows)


def _deterministic_splits():
    train = pd.DataFrame(
        {
            "tire_compound": ["HARD", "HARD", "SOFT", "SOFT"],
            "tire_age_laps": [1, 2, 1, 2],
            "next_lap_wear_increment": [0.5, 0.7, 0.9, 1.1],
        }
    )
    validation = pd.DataFrame(
        {
            "tire_compound": ["HARD", "HARD", "SOFT", "SOFT"],
            "tire_age_laps": [1, 2, 1, 2],
            "next_lap_wear_increment": [0.5, 0.7, 0.9, 1.1],
        }
    )
    return train, validation


def test_target_structure_reports_within_group_variation_entropy_and_variance():
    report = compute_target_structure(_variable_frame())

    assert report["target_unique_values"] == 10
    assert report["compound_age_group_count"] == 4
    assert report["groups_with_multiple_targets"] == 4
    assert report["groups_with_zero_sample_std"] == 0
    assert report["conditional_binned_entropy_bits"] > 0
    assert report["variance_explained_by_compound_and_age"] < 1.0
    assert "race_track_temp_c" in report["numeric_feature_correlations"]


def test_deterministic_compound_age_lookup_fails_research_guard():
    train, validation = _deterministic_splits()

    with pytest.raises(ValueError, match="determinism guard"):
        diagnose_target_determinism(train, validation)


def test_deterministic_demo_exception_is_explicit():
    train, validation = _deterministic_splits()

    report = diagnose_target_determinism(train, validation, deterministic_demo=True)

    assert report["status"] == "DETERMINISTIC_DEMO_EXCEPTION"
    assert report["compound_age_lookup"]["r2"] == 1.0


def test_final_holdout_cannot_be_used_by_development_diagnostic():
    train, validation = _deterministic_splits()

    with pytest.raises(ValueError, match="final holdout"):
        diagnose_target_determinism(train, validation, evaluation_split="holdout", deterministic_demo=True)