from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from f1_tyre.config import TARGET_NAME
from f1_tyre.data import build_model_features, build_target_frame
from f1_tyre.evaluation.baselines import fit_baseline_models, score_baselines
from f1_tyre.evaluation.explainability import grouped_permutation_importance
from f1_tyre.evaluation.splits import chronological_race_splits
from f1_tyre.features import prepare_feature_matrix
from f1_tyre.strict_feature_policy import FEATURE_METADATA


DEFAULT_DATASET = PROJECT_ROOT / "data" / "generated" / "authoritative_seed_20261004.csv"
EVALUATION_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
CORRECTED_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "corrected"
RANDOM_STATE = 42
TARGET_DEFINITION = (
    "Simulated wear increment realized during lap t+1 from race state available at the end of lap t."
)

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "ridge": {"alpha": 3.0},
    "random_forest": {
        "n_estimators": 180,
        "min_samples_leaf": 3,
        "max_features": 0.8,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "neural_network": {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": 0.001,
        "max_iter": 180,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 15,
        "batch_size": 512,
        "random_state": RANDOM_STATE,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _script_hash() -> str:
    return _sha256(Path(__file__).resolve())


def prepare_training_data(dataset_path: str | Path):
    source = Path(dataset_path)
    raw = pd.read_csv(source)
    labeled = build_target_frame(raw)
    prepared = prepare_feature_matrix(labeled)
    partitions, protocol = chronological_race_splits(prepared)
    feature_columns = sorted(
        column
        for column, metadata in FEATURE_METADATA.items()
        if column in prepared.columns
        and metadata["allowed"]
        and not metadata["lagged"]
        and not metadata["rolling"]
        and metadata["availability"] != "partition_only"
    )
    if not feature_columns:
        raise ValueError("No registered current-state features are available for training.")
    feature_frames = {
        name: build_model_features(frame, feature_columns)
        for name, frame in partitions.items()
    }
    return labeled, partitions, feature_frames, feature_columns, protocol


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def _make_estimator(model_name: str):
    config = MODEL_CONFIGS[model_name]
    if model_name == "ridge":
        return Ridge(**config)
    if model_name == "random_forest":
        return RandomForestRegressor(**config)
    if model_name == "xgboost":
        return XGBRegressor(**config)
    if model_name == "neural_network":
        return MLPRegressor(**config)
    raise ValueError(f"Unsupported corrected model: {model_name}")


def _make_pipeline(model_name: str, training_features: pd.DataFrame) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _build_preprocessor(training_features)),
            ("model", _make_estimator(model_name)),
        ]
    )


def _score(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    return {
        "r2": float(r2_score(actual_values, predicted_values)),
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "number_of_rows": int(len(actual_values)),
    }


def _development_paths(seed: int) -> tuple[Path, Path]:
    return (
        PROJECT_ROOT / "artifacts" / "models" / f"development_seed_{seed}",
        EVALUATION_DIR / f"development_model_metrics_seed_{seed}.json",
    )


def _fit_development(
    dataset_path: Path,
    seed: int,
    dataset_hash: str,
    labeled: pd.DataFrame,
    partitions: dict[str, pd.DataFrame],
    feature_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    train_features = feature_frames["train"]
    train_target = partitions["train"][TARGET_NAME].astype(float)
    validation_features = feature_frames["validation"]
    validation_target = partitions["validation"][TARGET_NAME].astype(float)
    development_dir, report_path = _development_paths(seed)
    development_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_name, model_config in MODEL_CONFIGS.items():
        pipeline = _make_pipeline(model_name, train_features)
        pipeline.fit(train_features, train_target)
        results[model_name] = {
            "validation": _score(validation_target, pipeline.predict(validation_features)),
            "model_config": model_config,
        }
        joblib.dump(
            {
                "model": pipeline,
                "feature_columns": feature_columns,
                "metadata": {
                    "phase": "development_only",
                    "dataset_sha256": dataset_hash,
                    "script_sha256": _script_hash(),
                    "model_config": model_config,
                    "train_races": protocol["race_ids"]["train"],
                    "validation_races": protocol["race_ids"]["validation"],
                    "holdout_races": protocol["race_ids"]["holdout"],
                    "preprocessing_fit_scope": "train only",
                },
            },
            development_dir / f"{model_name}.joblib",
        )

    report = {
        "status": "DEVELOPMENT_VALIDATION_ONLY",
        "dataset": {
            "type": "synthetic",
            "path": dataset_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": dataset_hash,
            "rows": int(len(labeled)),
            "races": int(labeled["race_id"].nunique()),
        },
        "evaluation_protocol": protocol,
        "preprocessing_fit_scope": "train only",
        "final_holdout_scored": False,
        "feature_columns": feature_columns,
        "models": results,
        "model_configs": MODEL_CONFIGS,
        "random_state": RANDOM_STATE,
        "seed_label": seed,
        "script_sha256": _script_hash(),
        "git_revision": _git_revision(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "tyre_state": [],
        "thermal_weather": [],
        "circuit_load": [],
        "pace_and_sectors": [],
        "fuel_and_energy": [],
        "traffic_and_position": [],
        "driver_and_team": [],
        "race_context": [],
    }
    for feature in feature_columns:
        lowered = feature.lower()
        if feature in {"tire_compound", "tire_age_laps", "tire_age_ratio", "tire_age_squared"}:
            group = "tyre_state"
        elif any(token in lowered for token in ("temp", "weather", "humidity", "wind", "grip", "track_status")):
            group = "thermal_weather"
        elif feature.startswith("circuit_"):
            group = "circuit_load"
        elif any(token in lowered for token in ("lap_time", "s1_time", "s2_time", "s3_time", "sector_total")):
            group = "pace_and_sectors"
        elif any(token in lowered for token in ("fuel", "ers_")):
            group = "fuel_and_energy"
        elif any(token in lowered for token in ("gap_", "position", "drs_")):
            group = "traffic_and_position"
        elif feature.startswith(("driver_", "team_")):
            group = "driver_and_team"
        else:
            group = "race_context"
        groups[group].append(feature)
    return {name: columns for name, columns in groups.items() if columns}


def _code_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": version("scikit-learn"),
        "xgboost": version("xgboost"),
    }


def _finalize_evaluation(dataset_path: Path, seed: int) -> dict[str, Any]:
    final_metrics_path = EVALUATION_DIR / "final_metrics.json"
    final_report_path = EVALUATION_DIR / "final_report.json"
    ledger_path = EVALUATION_DIR / f"final_evaluation_ledger_seed_{seed}.json"
    if final_metrics_path.exists() or final_report_path.exists() or ledger_path.exists():
        raise FileExistsError("Final evaluation already started; use a new dataset/seed rather than scoring holdout twice.")

    labeled, partitions, feature_frames, feature_columns, protocol = prepare_training_data(dataset_path)
    dataset_hash = _sha256(dataset_path)
    development_dir, development_report_path = _development_paths(seed)
    if not development_report_path.exists():
        raise FileNotFoundError("Run development validation first; finalization never fits or selects models.")
    development_report = json.loads(development_report_path.read_text(encoding="utf-8"))
    if development_report["dataset"]["sha256"] != dataset_hash:
        raise ValueError("Development models were fitted using a different dataset hash.")
    if development_report["script_sha256"] != _script_hash():
        raise ValueError("Training code changed after development; rerun validation before final evaluation.")
    if development_report["feature_columns"] != feature_columns:
        raise ValueError("Development model feature list differs from the final evaluation input.")

    pipelines = {}
    for model_name in MODEL_CONFIGS:
        development_artifact = joblib.load(development_dir / f"{model_name}.joblib")
        if development_artifact["metadata"]["dataset_sha256"] != dataset_hash:
            raise ValueError(f"Development artifact for {model_name} has a different dataset hash.")
        pipelines[model_name] = development_artifact["model"]

    # The ledger is written before the first holdout prediction to prevent accidental rescoring.
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(
            {
                "dataset_sha256": dataset_hash,
                "training_code_revision": _git_revision(),
                "final_holdout_scoring_started": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    validation_target = partitions["validation"][TARGET_NAME].astype(float)
    holdout_target = partitions["holdout"][TARGET_NAME].astype(float)
    validation_features = feature_frames["validation"]
    holdout_features = feature_frames["holdout"]
    model_metrics = []
    for model_name, pipeline in pipelines.items():
        validation_result = development_report["models"][model_name]["validation"]
        holdout_prediction = pipeline.predict(holdout_features)
        model_metrics.append(
            {
                "model": model_name,
                "validation": validation_result,
                "final_holdout": _score(holdout_target, holdout_prediction),
                "model_config": MODEL_CONFIGS[model_name],
            }
        )

    baselines = fit_baseline_models(partitions["train"], TARGET_NAME)
    baseline_validation = score_baselines(baselines, partitions["validation"], TARGET_NAME)
    baseline_holdout = score_baselines(
        baselines,
        partitions["holdout"],
        TARGET_NAME,
        evaluation_split="final_holdout",
        allow_final_holdout=True,
    )
    lookup_holdout = next(item for item in baseline_holdout if item["model"] == "CompoundAgeLookup")
    determinism_threshold = 0.98
    if lookup_holdout["r2"] >= determinism_threshold:
        raise ValueError(
            "Final compound-age lookup reached the determinism guard; stop and redesign the dataset before claims."
        )

    best_validation_model = max(
        development_report["models"],
        key=lambda name: development_report["models"][name]["validation"]["r2"],
    )
    importance = grouped_permutation_importance(
        pipelines[best_validation_model],
        validation_features,
        validation_target,
        _feature_groups(feature_columns),
        n_repeats=5,
        random_state=RANDOM_STATE,
    )
    importance_records = importance.to_dict(orient="records")
    code_revision = _git_revision()
    final_metrics: dict[str, Any] = {
        "schema_version": 1,
        "status": "FINAL_EVALUATION_COMPLETE",
        "dataset": {
            "type": "synthetic",
            "path": dataset_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": dataset_hash,
            "generation_seed": seed,
            "rows": int(len(labeled)),
            "races": int(labeled["race_id"].nunique()),
        },
        "target": {
            "name": TARGET_NAME,
            "definition": TARGET_DEFINITION,
            "prediction_moment": "end of lap t; features use state known no later than end of t",
        },
        "evaluation_protocol": protocol,
        "preprocessing_fit_scope": "train only",
        "hyperparameter_selection": "Fixed model configs; validation scores are reported; final holdout did not select models or features.",
        "holdout_evaluated_once": True,
        "training_code_revision": code_revision,
        "training_script_sha256": _script_hash(),
        "versions": _code_versions(),
        "target_structure_train": {
            "rows": int(len(partitions["train"])),
            "mean": float(partitions["train"][TARGET_NAME].mean()),
            "variance": float(partitions["train"][TARGET_NAME].var(ddof=1)),
        },
        "target_determinism_guard": {
            "threshold": determinism_threshold,
            "final_holdout_compound_age_lookup_r2": lookup_holdout["r2"],
            "passed": True,
        },
        "metrics": {"models": model_metrics, "baselines": baseline_validation + baseline_holdout},
        "explainability": {
            "method": "grouped_permutation_validation",
            "importance_model": best_validation_model,
            "metric": "increase_in_mae",
            "shap_computed": False,
            "results": importance_records,
        },
    }
    final_report: dict[str, Any] = {
        "title": "Synthetic Tyre-Wear Benchmark: Final Chronological Evaluation",
        "dataset_type": "synthetic",
        "research_scope": "Results apply only to the versioned synthetic simulator, not measured F1 races.",
        "target_definition": TARGET_DEFINITION,
        "prediction_moment": final_metrics["target"]["prediction_moment"],
        "race_counts": protocol["race_counts"],
        "metrics": final_metrics["metrics"],
        "compound_age_lookup_holdout_r2": lookup_holdout["r2"],
        "feature_importance_method": "grouped permutation on validation; predictive association only",
        "shap_computed": False,
        "limitations": [
            "The data are synthetic and do not establish real-world Formula 1 accuracy.",
            "Simulator coefficients and latent distributions are design assumptions, not empirically calibrated physical laws.",
            "Feature importance describes model dependence and is not causal evidence.",
        ],
    }

    CORRECTED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, pipeline in pipelines.items():
        joblib.dump(
            {
                "model": pipeline,
                "feature_columns": feature_columns,
                "target": TARGET_NAME,
                "metrics": next(item for item in model_metrics if item["model"] == model_name),
                "metadata": {
                    "model_name": model_name,
                    "dataset_type": "synthetic",
                    "dataset_sha256": dataset_hash,
                    "training_code_revision": code_revision,
                    "training_script_sha256": _script_hash(),
                    "train_races": protocol["race_ids"]["train"],
                    "validation_races": protocol["race_ids"]["validation"],
                    "holdout_races": protocol["race_ids"]["holdout"],
                    "feature_columns": feature_columns,
                    "model_config": MODEL_CONFIGS[model_name],
                    "preprocessing_fit_scope": "train only",
                    "target_name": TARGET_NAME,
                },
            },
            CORRECTED_MODEL_DIR / f"{model_name}.joblib",
        )

    (EVALUATION_DIR / "feature_importance_seed_20261004.json").write_text(
        json.dumps({"method": "grouped_permutation_validation", "results": importance_records}, indent=2),
        encoding="utf-8",
    )
    (EVALUATION_DIR / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (EVALUATION_DIR / "final_report.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    return final_metrics


def run(
    dataset_path: str | Path = DEFAULT_DATASET,
    seed: int = 20261004,
    finalize: bool = False,
) -> dict[str, Any]:
    source = Path(dataset_path)
    if not source.exists():
        raise FileNotFoundError(f"Generated dataset not found: {source}. Run the synthetic data generator first.")
    if finalize:
        return _finalize_evaluation(source, seed)

    labeled, partitions, feature_frames, feature_columns, protocol = prepare_training_data(source)
    dataset_hash = _sha256(source)
    return _fit_development(
        source,
        seed,
        dataset_hash,
        labeled,
        partitions,
        feature_frames,
        feature_columns,
        protocol,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train corrected models without accessing final holdout during development.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--finalize", action="store_true", help="Evaluate the reserved holdout once using completed development models.")
    args = parser.parse_args()
    report = run(args.dataset, args.seed, args.finalize)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()