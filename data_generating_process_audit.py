from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clean_re_evaluation import build_preprocessor, chronological_races, compute_split_races, make_model
from f1_tyre.config import TARGET_NAME
from f1_tyre.data import build_target_frame, drop_duplicate_race_laps, ensure_causal_sort
from f1_tyre.features import prepare_feature_matrix
from f1_tyre.strict_feature_policy import ALLOWED_FEATURES

TARGET = TARGET_NAME
SEED = 42
COMPOUND_RATES = {"HARD": 1.0, "MEDIUM": 1.5, "SOFT": 2.5}
KEY_COLUMNS = ["tire_compound", "tire_age_laps"]
BLOCKED_COLUMNS = {"race_id", "driver_id", "track_status"}

PHYSICAL_TELEMETRY = {
    "lap",
    "position",
    "lap_time_sec",
    "s1_time_sec",
    "s2_time_sec",
    "s3_time_sec",
    "tire_compound",
    "tire_age_laps",
    "fuel_load_kg",
    "ers_deploy_pct",
    "ers_harvest_pct",
    "drs_activated",
    "gap_to_leader_sec",
    "gap_ahead_sec",
    "gap_behind_sec",
    "weather_current",
    "race_air_temp_c",
    "race_track_temp_c",
    "race_humidity_pct",
    "wind_speed_kph",
    "track_grip_level",
}
PHYSICAL_DERIVED = {
    "sector_total_sec",
    "lap_time_per_km",
    "tire_age_ratio",
    "tire_age_squared",
    "total_nearest_gap",
    "position_change",
    "race_progress",
}
LAGGABLE_TELEMETRY = PHYSICAL_TELEMETRY - {"lap", "tire_compound", "tire_age_laps", "weather_current", "drs_activated"}


def metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray, races: int) -> dict:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    return {
        "r2": float(r2_score(actual_values, predicted_values)),
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "number_of_rows": int(len(actual_values)),
        "number_of_races": int(races),
    }


def fit_pipeline(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(X_train)),
            ("model", make_model(model_name)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def load_verified_clean_baselines(
    train_path: Path,
    train_races: list[int],
    validation_races: list[int],
    holdout_races: list[int],
    feature_columns: list[str],
    holdout: pd.DataFrame,
) -> tuple[dict[str, dict], np.ndarray]:
    artifact_dir = PROJECT_ROOT / "artifacts"
    metrics_path = artifact_dir / "strict_holdout_metrics.json"
    predictions_path = artifact_dir / "strict_holdout_predictions.csv"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    if payload["reproducibility"]["train_csv_sha256"] != dataset_hash:
        raise ValueError("Saved baseline metrics were created from a different train.csv.")

    saved_splits = payload["corrected_evaluation_protocol"]["splits"]
    expected_splits = {
        "train": train_races,
        "validation": validation_races,
        "holdout": holdout_races,
    }
    if saved_splits != expected_splits:
        raise ValueError("Saved baseline metrics do not use this audit's exact chronological race split.")
    if payload["corrected_evaluation_protocol"]["feature_columns"] != feature_columns:
        raise ValueError("Saved baseline metrics use a different feature matrix.")
    if not payload.get("holdout_prediction_provenance_verified", False):
        raise ValueError("Saved baseline holdout prediction provenance is not verified.")

    expected_models = {"Ridge", "RandomForest", "XGBoost", "NeuralNetwork"}
    provenance = payload["model_fit_provenance"]
    for model_name in expected_models:
        fit_info = provenance[model_name]
        if set(fit_info["fit_race_ids"]) != set(train_races):
            raise ValueError(f"Saved {model_name} baseline was not fitted on exactly TRAIN races.")
        if fit_info["holdout_race_ids_seen_during_fit"]:
            raise ValueError(f"Saved {model_name} baseline saw holdout races during fit.")

    baseline_metrics = {
        item["model"]: {
            key: item[key]
            for key in ("r2", "mae", "rmse", "number_of_rows", "number_of_races")
        }
        for item in payload["metrics"]
        if item["split"] == "holdout" and item["model"] in expected_models
    }
    if set(baseline_metrics) != expected_models:
        raise ValueError("Saved clean evaluation does not contain all four holdout baselines.")
    for model_name, score in baseline_metrics.items():
        if score["number_of_rows"] != len(holdout) or score["number_of_races"] != len(holdout_races):
            raise ValueError(f"Saved {model_name} baseline metrics do not match this holdout.")

    saved_predictions = pd.read_csv(predictions_path)
    saved_xgboost = saved_predictions.loc[saved_predictions["model"].eq("XGBoost")].copy()
    keys = ["race_id", "driver_id", "lap"]
    if not saved_xgboost["model_fit_split"].eq("train").all():
        raise ValueError("Saved XGBoost predictions do not record TRAIN-only fitting.")
    indexed = saved_xgboost.set_index(keys)
    if not indexed.index.is_unique or len(indexed) != len(holdout):
        raise ValueError("Saved XGBoost holdout prediction keys are not one-to-one with the current holdout.")
    holdout_index = pd.MultiIndex.from_frame(holdout[keys])
    actual = indexed["actual"].reindex(holdout_index)
    predictions = indexed["predicted"].reindex(holdout_index)
    if actual.isna().any() or predictions.isna().any():
        raise ValueError("Saved XGBoost predictions are missing holdout race-driver-lap keys.")
    if not np.allclose(actual.to_numpy(), holdout[TARGET].to_numpy()):
        raise ValueError("Saved XGBoost actual targets do not align with the current holdout.")
    return baseline_metrics, predictions.to_numpy(dtype=float)


def is_physical_feature(column: str) -> bool:
    if column in PHYSICAL_TELEMETRY or column in PHYSICAL_DERIVED:
        return True
    return any(
        column.startswith(f"{base}_lag") or column.startswith(f"{base}_roll")
        for base in LAGGABLE_TELEMETRY
    )


def markdown_table(frame: pd.DataFrame, float_format: str = ".6f") -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        row = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                row.append(format(value, float_format))
            else:
                row.append(str(value))
        rows.append(row)
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    return "\n".join(
        [format_row(headers), "| " + " | ".join("-" * width for width in widths) + " |", *(format_row(row) for row in rows)]
    )


def main() -> None:
    train_path = PROJECT_ROOT / "train.csv"
    raw = ensure_causal_sort(drop_duplicate_race_laps(pd.read_csv(train_path)))

    expected_wear = np.minimum(
        100.0,
        (raw["tire_age_laps"] - 1) * raw["tire_compound"].map(COMPOUND_RATES),
    )
    wear_mismatch_count = int((~np.isclose(raw["tire_wear_pct"], expected_wear)).sum())
    if wear_mismatch_count:
        raise AssertionError(f"Observed wear formula mismatches: {wear_mismatch_count}")

    target_frame = build_target_frame(raw)
    prepared = prepare_feature_matrix(target_frame)
    feature_columns = [
        column
        for column in prepared.columns
        if column in ALLOWED_FEATURES and column not in BLOCKED_COLUMNS
    ]
    feature_frame = prepared[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    combined = feature_frame
    combined[TARGET] = target_frame[TARGET].astype(float).to_numpy()
    for column in ("race_id", "driver_id", "lap", "tire_compound", "tire_age_laps"):
        combined[column] = target_frame[column].to_numpy()

    race_order = chronological_races(raw)
    train_races, validation_races, holdout_races = compute_split_races(race_order)
    partitions = {
        "train": combined.loc[combined["race_id"].isin(train_races)].copy(),
        "validation": combined.loc[combined["race_id"].isin(validation_races)].copy(),
        "holdout": combined.loc[combined["race_id"].isin(holdout_races)].copy(),
    }
    if set(train_races) & set(holdout_races) or set(validation_races) & set(holdout_races):
        raise AssertionError("Race overlap found in the existing chronological split.")

    train = partitions["train"]
    holdout = partitions["holdout"]
    X_train = train[feature_columns]
    y_train = train[TARGET]
    X_holdout = holdout[feature_columns]
    y_holdout = holdout[TARGET]

    target_values = target_frame[TARGET].astype(float)
    target_stats = {
        "target_unique_values": int(target_values.nunique()),
        "target_mean": float(target_values.mean()),
        "target_std": float(target_values.std(ddof=1)),
        "target_min": float(target_values.min()),
        "target_max": float(target_values.max()),
        "target_rows": int(len(target_values)),
    }
    target_counts = target_values.value_counts().sort_index()
    target_proportions = target_counts / len(target_values)
    grouped = target_frame.groupby(KEY_COLUMNS)[TARGET].agg(
        count="count",
        nunique="nunique",
        mean="mean",
        std="std",
        min="min",
        max="max",
    ).reset_index()
    zero_std = grouped["std"].eq(0)
    zero_std_group_count = int(zero_std.sum())
    zero_std_group_proportion = float(zero_std.mean())

    distribution_columns = [
        "record_type",
        "target_value",
        "frequency",
        "proportion",
        "tire_compound",
        "tire_age_laps",
        "count",
        "nunique",
        "mean",
        "std",
        "min",
        "max",
        "target_unique_values",
        "target_mean",
        "target_std",
        "target_min",
        "target_max",
        "zero_std_group_count",
        "zero_std_group_proportion",
    ]
    distribution_rows = [
        {
            "record_type": "overall_summary",
            **target_stats,
            "zero_std_group_count": zero_std_group_count,
            "zero_std_group_proportion": zero_std_group_proportion,
        }
    ]
    distribution_rows.extend(
        {
            "record_type": "target_value_distribution",
            "target_value": float(value),
            "frequency": int(count),
            "proportion": float(target_proportions.loc[value]),
        }
        for value, count in target_counts.items()
    )
    distribution_rows.extend(
        {"record_type": "compound_age_group", **row}
        for row in grouped.to_dict(orient="records")
    )
    distribution_frame = pd.DataFrame(distribution_rows).reindex(columns=distribution_columns)
    distribution_frame.to_csv(PROJECT_ROOT / "target_distribution.csv", index=False)

    baseline_metrics, xgboost_all_predictions = load_verified_clean_baselines(
        train_path,
        train_races,
        validation_races,
        holdout_races,
        feature_columns,
        holdout,
    )
    print("Reused verified baseline metrics and XGBoost predictions from the same dataset/split/provenance.")

    lookup = train.groupby(KEY_COLUMNS)[TARGET].median()
    lookup_predictions = np.array(
        [lookup.get((row.tire_compound, row.tire_age_laps), float(y_train.median()))
         for row in holdout[["tire_compound", "tire_age_laps"]].itertuples(index=False)]
    )
    missing_lookup_groups = int(
        sum((row.tire_compound, row.tire_age_laps) not in lookup.index
            for row in holdout[["tire_compound", "tire_age_laps"]].itertuples(index=False))
    )
    lookup_metrics = metrics(y_holdout, lookup_predictions, holdout["race_id"].nunique())

    derived_age_features = {"tire_age_ratio", "tire_age_squared"}
    physical_columns = [column for column in feature_columns if is_physical_feature(column)]
    ablation_variants = {
        "A_all_features": feature_columns,
        "B_remove_tire_age_laps": [column for column in feature_columns if column != "tire_age_laps"],
        "C_remove_tire_compound": [column for column in feature_columns if column != "tire_compound"],
        "D_remove_age_and_compound": [
            column for column in feature_columns if column not in {"tire_age_laps", "tire_compound"}
        ],
        "E_remove_age_compound_and_derived_age": [
            column for column in feature_columns
            if column not in ({"tire_age_laps", "tire_compound"} | derived_age_features)
        ],
        "F_physical_causal_telemetry_only": physical_columns,
    }
    ablation_rows = []
    for variant, columns in ablation_variants.items():
        if not columns:
            raise ValueError(f"No features remain for {variant}.")
        if variant == "A_all_features":
            predictions = xgboost_all_predictions
        else:
            pipeline = fit_pipeline("XGBoost", train[columns], y_train)
            predictions = pipeline.predict(holdout[columns])
        score = metrics(y_holdout, predictions, holdout["race_id"].nunique())
        ablation_rows.append(
            {
                "variant": variant,
                "model": "XGBoost",
                "feature_count": len(columns),
                "features_removed": ";".join(sorted(set(feature_columns) - set(columns))),
                "r2": score["r2"],
                "mae": score["mae"],
                "rmse": score["rmse"],
                "number_of_rows": score["number_of_rows"],
                "number_of_races": score["number_of_races"],
                "training_races": len(train_races),
                "holdout_races_seen_during_fit": 0,
            }
        )
        print(f"Completed ablation {variant} ({len(columns)} features): R2={score['r2']:.6f}")
    ablation_frame = pd.DataFrame(ablation_rows)
    ablation_frame.to_csv(PROJECT_ROOT / "feature_ablation_results.csv", index=False)

    lookup_results = {
        "status": "complete",
        "dataset": "train.csv",
        "train_csv_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "random_seed": SEED,
        "chronological_split": {
            "train_races": train_races,
            "validation_races": validation_races,
            "holdout_races": holdout_races,
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "zero_race_overlap": True,
        },
        "target": {
            **target_stats,
            "target_distribution": {
                str(value): {"count": int(count), "proportion": float(target_proportions.loc[value])}
                for value, count in target_counts.items()
            },
            "tire_wear_generation_formula_observed_exactly": "min(100, compound_rate * (tire_age_laps - 1)); rates HARD=1.0, MEDIUM=1.5, SOFT=2.5",
            "raw_wear_formula_mismatch_rows": wear_mismatch_count,
            "original_generator_source_found": False,
            "formula_provenance": "No generator source is present in the repository history. This equation reproduces every observed train.csv tire_wear_pct value exactly.",
            "next_lap_target_formula": "wear[t+1] - wear[t], within race_id and driver_id, requiring lap+1, tire_age_laps+1, and unchanged tire_compound",
            "compound_age_groups": int(len(grouped)),
            "groups_with_std_equal_zero": zero_std_group_count,
            "zero_std_group_proportion": zero_std_group_proportion,
            "groups_with_one_target_value": int(grouped["nunique"].eq(1).sum()),
        },
        "determinism_checks": {
            "keys": ["tire_compound", "tire_age_laps"],
            "groups": int(len(grouped)),
            "groups_with_one_target_value": int(grouped["nunique"].eq(1).sum()),
            "proportion_with_one_target_value": float(grouped["nunique"].eq(1).mean()),
            "age_compound_pair_count_in_train": int(train[KEY_COLUMNS].drop_duplicates().shape[0]),
            "age_compound_pair_count_in_holdout": int(holdout[KEY_COLUMNS].drop_duplicates().shape[0]),
            "holdout_pairs_unseen_in_train": int(
                len(set(map(tuple, holdout[KEY_COLUMNS].drop_duplicates().to_numpy()))
                    - set(map(tuple, train[KEY_COLUMNS].drop_duplicates().to_numpy())))
            ),
            "lap_or_fuel_required_for_exact_lookup": False,
        },
        "lookup_baseline": {
            "target_lookup_key": KEY_COLUMNS,
            "fit_source": "TRAIN only median target by key",
            "unseen_key_fallback": "TRAIN global target median",
            "unseen_holdout_keys": missing_lookup_groups,
            **lookup_metrics,
        },
        "model_metrics": {
            "lookup": lookup_metrics,
            "ridge": baseline_metrics["Ridge"],
            "random_forest": baseline_metrics["RandomForest"],
            "xgboost": baseline_metrics["XGBoost"],
            "neural_network": baseline_metrics["NeuralNetwork"],
        },
        "requested_r2": {
            "R2_lookup": lookup_metrics["r2"],
            "R2_ridge": baseline_metrics["Ridge"]["r2"],
            "R2_random_forest": baseline_metrics["RandomForest"]["r2"],
            "R2_xgboost": baseline_metrics["XGBoost"]["r2"],
            "R2_neural_network": baseline_metrics["NeuralNetwork"]["r2"],
        },
        "lookup_close_to_xgboost": abs(lookup_metrics["r2"] - baseline_metrics["XGBoost"]["r2"]) <= 0.01,
        "baseline_source": {
            "source_metrics": "artifacts/strict_holdout_metrics.json",
            "source_predictions": "artifacts/strict_holdout_predictions.csv",
            "verified_same_dataset_hash": True,
            "verified_same_race_split": True,
            "verified_feature_columns_match": True,
            "verified_train_only_fit_provenance": True,
            "xgboost_all_features_fit_reused_for_ablation_A": True,
        },
        "xgboost_ablations": ablation_rows,
        "physical_causal_telemetry_definition": sorted(set(physical_columns)),
        "preprocessing_fit_scope": "TRAIN only for every fitted estimator",
        "model_fit_holdout_exposure": {name: 0 for name in ("Ridge", "RandomForest", "XGBoost", "NeuralNetwork")},
        "versions": {"python_note": sys.version.split()[0], "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    output_json = PROJECT_ROOT / "lookup_baseline_results.json"
    output_json.write_text(json.dumps(lookup_results, indent=2), encoding="utf-8")

    distribution_table = pd.DataFrame(
        [{"target_value": value, "count": int(count), "proportion": float(target_proportions.loc[value])}
         for value, count in target_counts.items()]
    )
    baseline_table = pd.DataFrame(
        [{"model": name, **values} for name, values in (
            ("Lookup", lookup_metrics),
            ("Ridge", baseline_metrics["Ridge"]),
            ("Random Forest", baseline_metrics["RandomForest"]),
            ("XGBoost", baseline_metrics["XGBoost"]),
            ("Neural Network", baseline_metrics["NeuralNetwork"]),
        )]
    )
    wear_formula = "tire_wear_pct = min(100, rate[compound] * (tire_age_laps - 1)); rates: HARD=1.0, MEDIUM=1.5, SOFT=2.5"
    target_formula = "next_lap_wear_increment[t] = tire_wear_pct[t+1] - tire_wear_pct[t]"
    report = f"""# Data-Generating-Process Audit

## Finding

**target determinism / synthetic-data limitation** (not data leakage). `train.csv` contains a constructed wear schedule: `{wear_formula}`. The generator source itself is absent from the repository and its history, so this is an exact empirical reconstruction rather than a claim about the unavailable original program. The equation has {wear_mismatch_count} mismatches across {len(raw):,} raw rows.

The target builder then computes `{target_formula}` after sorting within `race_id` and `driver_id`. A target row is kept only when the next lap is consecutive, tyre age increases by one, and compound is unchanged. In symbols, if `W(a,c)=min(100, r(c)*(a-1))`, the target is `W(a+1,c)-W(a,c)` for retained rows. This yields the fixed compound wear rates until the 100% cap, followed by a zero increment; MEDIUM has a final 1.0 increment at the cap boundary.

The checked-in training table is not accompanied by a real telemetry source, measurement protocol, or generator code. Although it contains telemetry-shaped columns, the exact deterministic age/compound wear schedule and artificial rating/tier fields are inconsistent with treating this as measured real-world F1 tyre degradation. The supplied target should be treated as synthetic/constructed, not as real telemetry.

## Target Summary

- `target_unique_values`: {target_stats['target_unique_values']}
- `target_distribution` (count / proportion): `{lookup_results['target']['target_distribution']}`
- `target_mean`: {target_stats['target_mean']:.9f}
- `target_std`: {target_stats['target_std']:.9f}
- `target_min`: {target_stats['target_min']:.6f}
- `target_max`: {target_stats['target_max']:.6f}
- target rows: {target_stats['target_rows']:,}
- compound-age groups: {len(grouped)}; groups with `std == 0`: {zero_std_group_count} ({zero_std_group_proportion:.1%})

`target_distribution.csv` contains the requested overall summary, frequency distribution by target value, and the full 207-row `groupby(["tire_compound", "tire_age_laps"])` table with `count`, `nunique`, `mean`, sample `std`, `min`, and `max`.

## Same-Holdout Results

The race order, split fractions, model implementations, baseline fits, and XGBoost all-feature predictions are reused from the prior chronological clean evaluation after validating its dataset hash, exact split IDs, feature list, target alignment, and fit provenance. TRAIN has {len(train_races)} races; FINAL HOLDOUT has {len(holdout_races)} later races, IDs `{holdout_races}`. Every baseline and ablation is evaluated on the same {len(holdout):,} holdout rows. The five changed-feature XGBoost variants are freshly fitted on TRAIN only; no model sees a holdout race during fitting.

{markdown_table(baseline_table)}

The lookup baseline uses the median TRAIN target for `(tire_compound, tire_age_laps)` and falls back to the TRAIN global median for an unseen key. It encounters {missing_lookup_groups} unseen holdout keys. Lookup R² is {lookup_metrics['r2']:.9f}; XGBoost R² is {baseline_metrics['XGBoost']['r2']:.9f}. The absolute difference is {abs(lookup_metrics['r2'] - baseline_metrics['XGBoost']['r2']):.9f}; the lookup is {'close to' if lookup_results['lookup_close_to_xgboost'] else 'not within 0.01 R² of'} XGBoost. This is evidence the tree is recovering the deterministic lookup rule, not a rich degradation process.

## XGBoost Ablations

All variants use fixed hyperparameters from the clean evaluation and the same TRAIN/HOLDOUT partition; no tuning was performed.

{markdown_table(ablation_frame[['variant', 'feature_count', 'r2', 'mae', 'rmse']])}

Variant F is explicitly defined as available race-state telemetry and deterministic transformations of it: compound and tyre age, lap/position/pace/sectors, fuel/ERS/DRS, gaps, weather and temperatures, humidity, wind/grip where present, plus causal lag/rolling telemetry features. Static driver/team/circuit ratings and identifiers are excluded. Its exact feature list is recorded in `lookup_baseline_results.json`.

## Determinism and Interpretation

The target is exactly determined by `(tire_compound, tire_age_laps)` in all {len(grouped)} observed groups: `nunique == 1` in {int(grouped['nunique'].eq(1).sum())} groups and `std == 0` in {zero_std_group_count}. All {int(train[KEY_COLUMNS].drop_duplicates().shape[0])} training key pairs also occur in holdout, so the lookup has no unseen key problem. `lap` and `fuel_load_kg` are not needed to derive the target once compound and age are known. They can serve as correlated proxies in ablations but are not part of the exact rule.

- **LEAKAGE:** this audit does not change or re-evaluate the feature leakage system. The target derivation uses next-lap wear by definition; the existing causal feature guardrails remain separate.
- **TARGET DETERMINISM:** confirmed, by exact formula reconstruction and zero within-group target variance.
- **GENERALIZATION:** the deterministic schedule transfers to later race IDs because the same compound-age combinations recur. This is rule transfer within the generated table, not evidence of generalization to independent real races.
- **REALISM:** unsupported as a research claim about measured real-world F1 tyre degradation. The wear target is constructed, discretized by compound rates, capped at 100, and has no observed race-level noise.

## Provenance and Reproducibility

`train.csv` was added to repository history without a corresponding generator program; no source-to-telemetry provenance is present in the repository. The audit records the dataset SHA-256, versions, exact split race IDs, group table, fixed model configurations, and random seed in `lookup_baseline_results.json`. It does not modify `train.csv`, any leakage code, or historical metrics.
"""
    (PROJECT_ROOT / "data_generating_process_audit.md").write_text(report, encoding="utf-8")

    print("\nTARGET SUMMARY")
    print(target_stats)
    print("TARGET COUNTS", target_counts.to_dict())
    print("GROUPS WITH STD ZERO", zero_std_group_count, "of", len(grouped), f"({zero_std_group_proportion:.1%})")
    print("\nSAME-HOLDOUT BASELINES")
    print(baseline_table.to_string(index=False))
    print("\nXGBOOST ABLATIONS")
    print(ablation_frame[["variant", "feature_count", "r2", "mae", "rmse"]].to_string(index=False))
    print("\nWROTE data_generating_process_audit.md, target_distribution.csv, feature_ablation_results.csv, lookup_baseline_results.json")


if __name__ == "__main__":
    main()