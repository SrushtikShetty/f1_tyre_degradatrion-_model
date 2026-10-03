import pandas as pd
import pytest

from data_generation.generate_dataset import generate_dataset, load_config
from evaluation.compare_baselines import run as report_baselines
from f1_tyre.evaluation.baselines import fit_baseline_models, score_baselines


def _frame(compounds, ages, targets):
    return pd.DataFrame(
        {
            "tire_compound": compounds,
            "tire_age_laps": ages,
            "next_lap_wear_increment": targets,
        }
    )


def test_baselines_fit_only_on_train_and_fallback_for_unseen_keys():
    train = _frame(["HARD", "HARD", "SOFT", "SOFT"], [1, 1, 1, 2], [1.0, 3.0, 2.0, 4.0])
    validation = _frame(["HARD", "MEDIUM"], [1, 7], [2.0, 5.0])

    baselines = fit_baseline_models(train)
    results = score_baselines(baselines, validation)
    compound_age = baselines["CompoundAgeLookup"]
    predictions, unseen = compound_age.predict(validation)

    assert predictions.tolist() == [2.0, 2.5]
    assert unseen == 1
    assert {record["split"] for record in results} == {"validation"}
    assert {record["metric_scope"] for record in results} == {"baseline_validation"}
    assert next(record for record in results if record["model"] == "CompoundAgeLookup")["unseen_lookup_keys"] == 1


def test_compound_only_lookup_uses_training_group_medians():
    train = _frame(["HARD", "HARD", "SOFT"], [1, 2, 1], [1.0, 3.0, 5.0])
    validation = _frame(["HARD", "SOFT"], [20, 1], [0.0, 0.0])

    baselines = fit_baseline_models(train)
    predictions, unseen = baselines["CompoundOnly"].predict(validation)

    assert predictions.tolist() == [2.0, 5.0]
    assert unseen == 0


def test_global_mean_baseline_uses_training_mean():
    train = _frame(["HARD", "SOFT"], [1, 1], [1.0, 5.0])
    validation = _frame(["MEDIUM", "HARD"], [8, 2], [0.0, 0.0])

    baselines = fit_baseline_models(train)
    predictions, _ = baselines["GlobalMean"].predict(validation)

    assert predictions.tolist() == [3.0, 3.0]


def test_final_holdout_baseline_scoring_requires_explicit_finalization():
    train = _frame(["HARD", "SOFT"], [1, 1], [1.0, 5.0])
    holdout = _frame(["HARD", "SOFT"], [2, 2], [2.0, 4.0])
    baselines = fit_baseline_models(train)

    with pytest.raises(ValueError, match="explicit final evaluation authorization"):
        score_baselines(baselines, holdout, evaluation_split="final_holdout")

    metrics = score_baselines(
        baselines,
        holdout,
        evaluation_split="final_holdout",
        allow_final_holdout=True,
    )
    assert {item["split"] for item in metrics} == {"final_holdout"}


def test_baseline_report_records_hash_and_does_not_score_holdout(tmp_path):
    config = load_config()
    config.update({"races": 6, "drivers": 4, "laps_per_race": 9})
    dataset = generate_dataset(config, seed=54)
    dataset_path = tmp_path / "synthetic.csv"
    report_path = tmp_path / "evaluation" / "baselines.json"
    dataset.to_csv(dataset_path, index=False)

    report = report_baselines(dataset_path, report_path, seed=54)

    assert report["final_holdout_scored"] is False
    assert report["dataset"]["seed"] == 54
    assert len(report["dataset"]["sha256"]) == 64
    assert {item["split"] for item in report["metrics"]} == {"validation"}
    assert report_path.exists()