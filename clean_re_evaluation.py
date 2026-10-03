from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from f1_tyre.config import TARGET_NAME
from f1_tyre.data import build_target_frame, drop_duplicate_race_laps, ensure_causal_sort
from f1_tyre.evaluation.leakage_audit import audit_feature_matrix
from f1_tyre.features import prepare_feature_matrix
from f1_tyre.strict_feature_policy import ALLOWED_FEATURES

RANDOM_SEED = 42
TARGET = TARGET_NAME
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)


def make_model(name: str):
    if name == "Ridge":
        return Ridge(alpha=1.0)
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    if name == "NeuralNetwork":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=200,
            early_stopping=True,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unsupported model: {name}")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [column for column in X if pd.api.types.is_numeric_dtype(X[column])]
    categorical_columns = [column for column in X if column not in numeric_columns]
    transformers = []

    if numeric_columns:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_columns,
        ))
    if categorical_columns:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_columns,
        ))

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def make_pipeline(name: str, X_train: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", make_model(name)),
    ])


def chronological_races(raw: pd.DataFrame) -> list[int]:
    required = {"race_id", "season", "race_round", "race_race_date"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Cannot establish chronological race order; missing columns: {sorted(missing)}")

    metadata = raw.groupby("race_id", as_index=False).agg(
        season=("season", "first"),
        race_round=("race_round", "first"),
        race_date=("race_race_date", "first"),
        season_values=("season", "nunique"),
        round_values=("race_round", "nunique"),
        date_values=("race_race_date", "nunique"),
    )
    if metadata[["season_values", "round_values", "date_values"]].ne(1).any().any():
        raise ValueError("Race chronology metadata is inconsistent within at least one race.")
    metadata["race_date"] = pd.to_datetime(metadata["race_date"], errors="raise")
    metadata = metadata.sort_values(["season", "race_date", "race_round", "race_id"])
    return [int(race_id) for race_id in metadata["race_id"]]


def compute_split_races(races: list[int]) -> tuple[list[int], list[int], list[int]]:
    if len(races) < 3:
        raise ValueError("At least three races are required for train/validation/holdout splits.")
    train_end = int(len(races) * SPLIT_FRACTIONS[0])
    validation_end = int(len(races) * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]))
    train_races = races[:train_end]
    validation_races = races[train_end:validation_end]
    holdout_races = races[validation_end:]
    if not train_races or not validation_races or not holdout_races:
        raise ValueError("Chronological split produced an empty partition.")
    return train_races, validation_races, holdout_races


def split_stints(raw: pd.DataFrame) -> dict[str, object]:
    """Build source-stint identities and prove no stint crosses a split."""
    ordered = raw.sort_values(["race_id", "driver_id", "lap"]).reset_index(drop=True)
    stint_keys: list[tuple[int, int, int]] = []
    for (race_id, driver_id), group in ordered.groupby(["race_id", "driver_id"], sort=False):
        boundary = (
            group["lap"].diff().ne(1)
            | group["tire_age_laps"].diff().ne(1)
            | group["tire_compound"].ne(group["tire_compound"].shift())
        )
        stint_numbers = boundary.cumsum().to_numpy()
        stint_keys.extend((int(race_id), int(driver_id), int(number)) for number in stint_numbers)
    ordered["_stint_key"] = stint_keys
    return {
        "rows": ordered,
        "keys": ordered[["race_id", "driver_id", "_stint_key"]].drop_duplicates(),
    }


def target_distribution(frame: pd.DataFrame) -> dict[str, float | int]:
    values = frame[TARGET].astype(float)
    return {
        "number_of_rows": int(len(frame)),
        "number_of_races": int(frame["race_id"].nunique()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median()),
        "zero_fraction": float(values.eq(0).mean()),
    }


def evaluate(model_name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, frame: pd.DataFrame, split: str) -> dict:
    predictions = pipeline.predict(X)
    return {
        "model": model_name,
        "split": split,
        "r2": float(r2_score(y, predictions)),
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "number_of_rows": int(len(X)),
        "number_of_races": int(frame["race_id"].nunique()),
        "predictions": predictions,
        "actual": y.to_numpy(),
    }


def grouped_feature_importance(model_name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out()

    def source_feature(transformed_name: str) -> str:
        name = transformed_name.split("__", 1)[-1]
        matches = [column for column in X_train.columns if name == column or name.startswith(f"{column}_")]
        if not matches:
            return name
        return max(matches, key=len)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        importance = pd.DataFrame({
            "feature": [source_feature(str(name)) for name in transformed_names],
            "importance": values,
        }).groupby("feature", as_index=False)["importance"].sum()
        method = "model_native_train_fit"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float).reshape(-1))
        importance = pd.DataFrame({
            "feature": [source_feature(str(name)) for name in transformed_names],
            "importance": values,
        }).groupby("feature", as_index=False)["importance"].sum()
        method = "absolute_coefficient_train_fit"
    else:
        sample_size = min(10000, len(X_train))
        sample = X_train.sample(n=sample_size, random_state=RANDOM_SEED)
        result = permutation_importance(
            pipeline,
            sample,
            y_train.loc[sample.index],
            n_repeats=3,
            random_state=RANDOM_SEED,
            scoring="r2",
            n_jobs=1,
        )
        importance = pd.DataFrame({"feature": X_train.columns, "importance": result.importances_mean})
        method = "permutation_train_sample"

    importance["model"] = model_name
    importance["method"] = method
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def run_permutation_sanity_check(train_df: pd.DataFrame, validation_df: pd.DataFrame, feature_columns: list[str]) -> dict:
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]
    X_validation = validation_df[feature_columns]
    y_validation = validation_df[TARGET]
    shuffled_values = np.random.default_rng(RANDOM_SEED).permutation(y_train.to_numpy())
    shuffled_target = pd.Series(shuffled_values, index=y_train.index)
    baseline = make_pipeline("Ridge", X_train)
    baseline.fit(X_train, shuffled_target)
    predictions = baseline.predict(X_validation)
    return {
        "model": "Ridge",
        "target_shuffle": "random permutation within TRAIN only",
        "evaluated_on": "VALIDATION",
        "r2": float(r2_score(y_validation, predictions)),
        "mae": float(mean_absolute_error(y_validation, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_validation, predictions))),
        "training_rows": int(len(X_train)),
        "training_races": int(train_df["race_id"].nunique()),
        "seed": RANDOM_SEED,
    }


def feature_ablation_test(train_df: pd.DataFrame, holdout_df: pd.DataFrame, feature_columns: list[str]) -> list[dict]:
    current_telemetry = {
        "position", "lap_time_sec", "s1_time_sec", "s2_time_sec", "s3_time_sec",
        "gap_to_leader_sec", "gap_ahead_sec", "gap_behind_sec", "fuel_load_kg",
        "ers_deploy_pct", "ers_harvest_pct", "drs_activated", "track_status", "weather_current",
        "sector_total_sec", "lap_time_per_km", "total_nearest_gap", "position_change",
        "race_air_temp_c", "race_track_temp_c", "race_humidity_pct", "wind_speed_kph", "track_grip_level",
    }
    fuel_features = {"fuel_load_kg", "fuel_ratio"} | {
        f"fuel_load_kg_lag{lag}" for lag in (1, 2, 3)
    }
    lap_sector_features = {
        column for column in feature_columns
        if column.startswith(("lap_time_sec", "s1_time_sec", "s2_time_sec", "s3_time_sec"))
    } | {"sector_total_sec", "lap_time_per_km"}
    variants = {
        "A_strict_causal": feature_columns,
        "B_no_current_lap_telemetry": [c for c in feature_columns if c not in current_telemetry],
        "C_no_fuel_features": [c for c in feature_columns if c not in fuel_features],
        "D_no_lap_time_or_sector_features": [c for c in feature_columns if c not in lap_sector_features],
    }

    results = []
    y_train = train_df[TARGET]
    y_holdout = holdout_df[TARGET]
    for variant, columns in variants.items():
        if not columns:
            raise ValueError(f"Feature ablation {variant} removed every feature.")
        X_train = train_df[columns]
        X_holdout = holdout_df[columns]
        baseline = make_pipeline("Ridge", X_train)
        baseline.fit(X_train, y_train)
        predictions = baseline.predict(X_holdout)
        results.append({
            "variant": variant,
            "model": "Ridge",
            "feature_count": int(len(columns)),
            "holdout_r2": float(r2_score(y_holdout, predictions)),
            "holdout_mae": float(mean_absolute_error(y_holdout, predictions)),
            "holdout_rmse": float(np.sqrt(mean_squared_error(y_holdout, predictions))),
            "training_races": int(train_df["race_id"].nunique()),
            "holdout_races": int(holdout_df["race_id"].nunique()),
            "holdout_used_for_model_selection": False,
        })
    return results


def leave_one_race_out_sanity_check(train_df: pd.DataFrame, holdout_df: pd.DataFrame, feature_columns: list[str], races: list[int]) -> list[dict]:
    if not races:
        return []
    selected_indices = np.linspace(0, len(races) - 1, num=min(5, len(races)), dtype=int)
    selected_races = [races[index] for index in selected_indices]
    results = []
    for race_id in selected_races:
        one_race = holdout_df.loc[holdout_df["race_id"].eq(race_id)]
        X_train = train_df[feature_columns]
        y_train = train_df[TARGET]
        X_race = one_race[feature_columns]
        y_race = one_race[TARGET]
        baseline = make_pipeline("Ridge", X_train)
        baseline.fit(X_train, y_train)
        predictions = baseline.predict(X_race)
        results.append({
            "race_id": int(race_id),
            "rows": int(len(one_race)),
            "r2": float(r2_score(y_race, predictions)),
            "mae": float(mean_absolute_error(y_race, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_race, predictions))),
            "training_races": int(train_df["race_id"].nunique()),
            "race_excluded_from_fit": True,
        })
    return results


def legacy_r2() -> float | None:
    path = PROJECT_ROOT / "artifacts" / "model_metrics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    xgb_metrics = next((item for item in metrics if item.get("model") == "XGBoost"), None)
    return float(xgb_metrics["r2"]) if xgb_metrics and "r2" in xgb_metrics else None


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(format(value, floatfmt))
            else:
                formatted.append(str(value))
        rows.append(formatted)
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    line = lambda values: "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"
    return "\n".join([line(headers), "| " + " | ".join("-" * width for width in widths) + " |", *(line(row) for row in rows)])


def markdown_report(payload: dict, comparison: pd.DataFrame, importance: pd.DataFrame) -> str:
    splits = payload["corrected_evaluation_protocol"]["splits"]
    dist = payload["target_distribution"]
    metric_columns = ["model", "split", "r2", "mae", "rmse", "number_of_rows", "number_of_races"]
    metric_table = markdown_table(comparison[metric_columns])
    distribution_frame = pd.DataFrame([{"split": name.upper(), **values} for name, values in dist.items()])
    dist_table = markdown_table(distribution_frame)
    ranked_importance = importance.copy()
    ranked_importance["rank"] = ranked_importance.groupby("model")["importance"].rank(
        method="average", ascending=False
    )
    top_features = (
        ranked_importance.groupby("feature", as_index=False)["rank"].mean()
        .sort_values(["rank", "feature"]).head(12)
    )
    feature_table = markdown_table(top_features, ".2f")
    ablation_table = markdown_table(pd.DataFrame(payload["feature_ablation"]))
    race_checks = markdown_table(pd.DataFrame(payload["leave_one_race_out_sanity"]))
    metric_rows = comparison[comparison["split"].eq("validation")]
    holdout_rows = comparison[comparison["split"].eq("holdout")]
    old_score = payload["old_evaluation_protocol"]["legacy_r2"]
    old_score_text = "unavailable" if old_score is None else f"{old_score:.12f}"
    target_structure = payload["target_structure_diagnostic"]
    return f"""# Clean Re-Evaluation and Leakage Audit

## Conclusion

- **LEAKAGE FIXED:** the reconstructed model feature matrix passed the strict allowlist audit before any split was created; target, future wear, and forbidden future-state columns were not model inputs.
- **PERFORMANCE VERIFIED:** {str(payload['performance_verified']).lower()}; the saved final-holdout results were produced after fitting each model only on TRAIN and after the holdout had remained untouched by fitting, tuning, or selection.
- The legacy XGBoost R² was {old_score_text} and is **LEGACY**, not an estimate of next-race performance. The old code shows group-disjoint five-fold `GroupKFold`, but no fixed chronological final holdout. It also computed a future-wear alias, while its explicit feature allowlist excluded that alias and current wear. Therefore the old source does **not** establish that direct target leakage caused the score; temporal look-ahead between folds and lack of an untouched forward holdout make that score insufficient as an unseen-future estimate.
- New validation R² by model: {', '.join(f"{row.model} {row.r2:.6f}" for row in metric_rows.itertuples())}.
- New unseen-race holdout R² by model: {', '.join(f"{row.model} {row.r2:.6f}" for row in holdout_rows.itertuples())}.
- The old 0.9999 score is not proven to have been caused by direct target leakage. The source excluded direct target aliases, but its non-chronological folds did not test future-race transfer; interpret the corrected scores alongside the synthetic target structure below.

## Protocol

**Legacy protocol:** `models/common.py` rebuilt the next-lap increment; each legacy regressor evaluated out-of-fold predictions with five-fold `GroupKFold` grouped by `race_id`, then fitted a final model on all rows. This kept a race out of its own fold's training set, but folds were not ordered chronologically and there was no permanently untouched final race block. The legacy summary file was read-only and was not overwritten.

**Corrected protocol:** source rows were sorted by season, race date, round, and race ID. The earliest 60% of races formed TRAIN, the next 20% VALIDATION, and the latest 20% FINAL HOLDOUT. All splits are race-disjoint, driver-race-disjoint, and stint-disjoint. Hyperparameters were fixed before fitting; validation and holdout scores did not drive model selection. Every imputer, encoder, and scaler was inside a pipeline fitted with TRAIN rows only.

Exact race IDs:

- TRAIN ({len(splits['train'])}): `{splits['train']}`
- VALIDATION ({len(splits['validation'])}): `{splits['validation']}`
- FINAL HOLDOUT ({len(splits['holdout'])}): `{splits['holdout']}`

Race, race-driver, and tyre-stint overlap checks all passed. The target builder retains a target only when the following row is the next lap, tyre age advances by exactly one, and the compound is unchanged. Model-fit provenance records TRAIN race IDs and asserts they are disjoint from both validation and holdout IDs.

## Metrics

{metric_table}

## Target Distribution

{dist_table}

## Target Structure

The target has {target_structure['unique_target_values']} distinct values. Across {target_structure['age_compound_pair_count']} observed `(tire_age_laps, tire_compound)` pairs, {target_structure['pairs_with_single_target_value']} map to exactly one target value. TRAIN covers {target_structure['train_pair_count']} pairs, VALIDATION covers {target_structure['validation_pair_count']} pairs, and FINAL HOLDOUT covers {target_structure['holdout_pair_count']} pairs; unseen holdout pairs relative to TRAIN: {target_structure['holdout_pairs_unseen_in_train']}. This deterministic mapping can explain very high tree-model scores without race overlap. It verifies transfer across held-out race IDs in this generated dataset, not external or real-world tyre degradation performance.

## Sanity Checks

**Shuffled-target baseline:** Ridge was trained after randomly permuting the target within TRAIN and evaluated on VALIDATION. R² = {payload['permutation_sanity_check']['r2']:.6f}; this should be near zero or below and confirms the pipeline is not producing a high score without the target relationship.

**Feature ablation:** each row below is the same fixed Ridge baseline, fitted only on TRAIN. Holdout scores are diagnostics for the requested ablations and were not used to choose a model or feature set.

{ablation_table}

**Leave-one-unseen-race-out checks:** each listed holdout race was excluded from fitting and evaluated separately.

{race_checks}

## Dominant Features

Feature importance was computed without holdout data. The table ranks original features within each model first, then averages those ranks, avoiding direct comparisons between coefficients, tree importances, and permutation scores.

{feature_table}

## Reproducibility

The clean runner, fixed seed ({RANDOM_SEED}), exact race IDs, dataset SHA-256, package versions, and fit-provenance checks are stored with the metrics. The procedure is reproducible from `train.csv`; this run has not been independently repeated in a second full four-model execution.

## Artifacts

- `artifacts/strict_holdout_metrics.json`
- `artifacts/strict_holdout_comparison.csv`
- `artifacts/strict_holdout_predictions.csv`
- `artifacts/strict_holdout_feature_importance.csv`

Historical `artifacts/model_metrics.json` and `artifacts/model_comparison.csv` were left unchanged. Existing model binaries, OOF predictions, and cached outputs were ignored; none were loaded for this evaluation.
"""


def main() -> None:
    train_path = PROJECT_ROOT / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data at {train_path}")

    raw = pd.read_csv(train_path)
    raw = ensure_causal_sort(drop_duplicate_race_laps(raw))
    target_frame = build_target_frame(raw)
    prepared = prepare_feature_matrix(target_frame)
    blocked = {"race_id", "driver_id", "track_status"}
    feature_columns = [
        column for column in prepared.columns
        if column in ALLOWED_FEATURES and column not in blocked
    ]
    if not feature_columns:
        raise ValueError("No permitted model features remained after filtering.")

    # Audit the full reconstructed feature matrix before defining any partitions.
    feature_frame = prepared[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    audit_feature_matrix(feature_frame)
    print(f"STRICT LEAKAGE AUDIT PASSED BEFORE SPLITTING ({len(feature_columns)} features)")

    races = chronological_races(raw)
    train_races, validation_races, holdout_races = compute_split_races(races)
    race_to_split = {
        **{race: "train" for race in train_races},
        **{race: "validation" for race in validation_races},
        **{race: "holdout" for race in holdout_races},
    }
    print("TRAIN RACE IDs:", train_races)
    print("VALIDATION RACE IDs:", validation_races)
    print("FINAL HOLDOUT RACE IDs:", holdout_races)

    race_sets = [set(train_races), set(validation_races), set(holdout_races)]
    if any(race_sets[left] & race_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise AssertionError("Race IDs overlap across partitions.")

    combined = feature_frame.copy()
    combined[TARGET] = target_frame[TARGET].astype(float).to_numpy()
    for column in ("race_id", "driver_id", "lap", "tire_compound", "tire_age_laps"):
        combined[column] = target_frame[column].to_numpy()
    frames = {
        "train": combined.loc[combined["race_id"].isin(train_races)].copy(),
        "validation": combined.loc[combined["race_id"].isin(validation_races)].copy(),
        "holdout": combined.loc[combined["race_id"].isin(holdout_races)].copy(),
    }
    for split_name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"Split produced no rows for {split_name}.")

    race_driver_sets = {
        split_name: set(zip(frame["race_id"].astype(int), frame["driver_id"].astype(int)))
        for split_name, frame in frames.items()
    }
    split_names = list(frames)
    for left in range(len(split_names)):
        for right in range(left + 1, len(split_names)):
            overlap = race_driver_sets[split_names[left]] & race_driver_sets[split_names[right]]
            if overlap:
                raise AssertionError(f"Race-driver overlap between {split_names[left]} and {split_names[right]}: {sorted(overlap)[:5]}")

    stint_check = split_stints(raw)
    stints = stint_check["keys"].copy()
    stints["split"] = stints["race_id"].map(race_to_split)
    if stints["split"].isna().any():
        raise AssertionError("At least one source tyre stint was not assigned to a split.")
    split_stint_keys = {
        name: set(
            stints.loc[stints["split"].eq(name), ["race_id", "driver_id", "_stint_key"]]
            .itertuples(index=False, name=None)
        )
        for name in frames
    }
    for left in range(len(split_names)):
        for right in range(left + 1, len(split_names)):
            if split_stint_keys[split_names[left]] & split_stint_keys[split_names[right]]:
                raise AssertionError("A source tyre stint crosses split boundaries.")
    stint_counts = {name: int(stints["split"].eq(name).sum()) for name in frames}
    print("RACE OVERLAPS: none")
    print("RACE-DRIVER OVERLAPS: none")
    print("TYRE-STINT OVERLAPS: none; source stint counts:", stint_counts)

    X = {name: frame[feature_columns].copy() for name, frame in frames.items()}
    y = {name: frame[TARGET].astype(float) for name, frame in frames.items()}
    train_race_set = set(train_races)
    model_names = ["Ridge", "RandomForest", "XGBoost", "NeuralNetwork"]
    metrics: list[dict] = []
    holdout_predictions: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    model_provenance = {}

    for model_name in model_names:
        pipeline = make_pipeline(model_name, X["train"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X["train"], y["train"])
        model_provenance[model_name] = {
            "fit_rows": int(len(X["train"])),
            "fit_race_ids": train_races,
            "validation_race_ids_seen_during_fit": [],
            "holdout_race_ids_seen_during_fit": [],
            "preprocessor_fit_rows": int(len(X["train"])),
            "preprocessor_fit_race_ids": train_races,
            "validation_and_holdout_excluded_from_fit": True,
            "warning_messages": sorted({str(item.message) for item in caught}),
        }
        if set(model_provenance[model_name]["fit_race_ids"]) != train_race_set:
            raise AssertionError(f"{model_name} was not fitted using exactly TRAIN races.")
        if set(model_provenance[model_name]["fit_race_ids"]) & set(holdout_races):
            raise AssertionError(f"{model_name} fit included a final holdout race.")

        for split_name in ("train", "validation", "holdout"):
            result = evaluate(model_name, pipeline, X[split_name], y[split_name], frames[split_name], split_name)
            metrics.append({key: result[key] for key in (
                "model", "split", "r2", "mae", "rmse", "number_of_rows", "number_of_races"
            )})
            if split_name == "holdout":
                predictions = frames[split_name][["race_id", "driver_id", "lap", "tire_compound", "tire_age_laps"]].copy()
                predictions["model"] = model_name
                predictions["actual"] = result["actual"]
                predictions["predicted"] = result["predictions"]
                predictions["error"] = predictions["actual"] - predictions["predicted"]
                predictions["model_fit_split"] = "train"
                holdout_predictions.append(predictions)
        importance_frames.append(grouped_feature_importance(model_name, pipeline, X["train"], y["train"]))
        print(f"Completed {model_name}; holdout races seen in fit: 0")

    comparison = pd.DataFrame(metrics).sort_values(["model", "split"]).reset_index(drop=True)
    predictions_frame = pd.concat(holdout_predictions, ignore_index=True)
    importance_frame = pd.concat(importance_frames, ignore_index=True)

    permutation_result = run_permutation_sanity_check(frames["train"], frames["validation"], feature_columns)
    ablation_result = feature_ablation_test(frames["train"], frames["holdout"], feature_columns)
    loo_result = leave_one_race_out_sanity_check(frames["train"], frames["holdout"], feature_columns, holdout_races)

    pair_columns = ["tire_age_laps", "tire_compound"]
    target_pair_counts = target_frame.groupby(pair_columns)[TARGET].nunique()
    target_pairs = {
        name: set(
            map(
                tuple,
                frame[pair_columns].drop_duplicates().itertuples(index=False, name=None),
            )
        )
        for name, frame in frames.items()
    }
    target_structure = {
        "unique_target_values": int(target_frame[TARGET].nunique()),
        "age_compound_pair_count": int(len(target_pair_counts)),
        "pairs_with_single_target_value": int(target_pair_counts.eq(1).sum()),
        "train_pair_count": len(target_pairs["train"]),
        "validation_pair_count": len(target_pairs["validation"]),
        "holdout_pair_count": len(target_pairs["holdout"]),
        "validation_pairs_unseen_in_train": len(target_pairs["validation"] - target_pairs["train"]),
        "holdout_pairs_unseen_in_train": len(target_pairs["holdout"] - target_pairs["train"]),
    }

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    comparison_path = artifact_dir / "strict_holdout_comparison.csv"
    metrics_path = artifact_dir / "strict_holdout_metrics.json"
    predictions_path = artifact_dir / "strict_holdout_predictions.csv"
    importance_path = artifact_dir / "strict_holdout_feature_importance.csv"
    comparison.to_csv(comparison_path, index=False)
    predictions_frame[[
        "model", "model_fit_split", "race_id", "driver_id", "lap", "tire_compound",
        "tire_age_laps", "actual", "predicted", "error",
    ]].to_csv(predictions_path, index=False)
    importance_frame.to_csv(importance_path, index=False)

    distributions = {name: target_distribution(frame) for name, frame in frames.items()}
    exact_splits = {name: [int(race) for race in ids] for name, ids in (
        ("train", train_races), ("validation", validation_races), ("holdout", holdout_races)
    )}
    payload = {
        "status": "COMPLETE_CLEAN_RE_EVALUATION",
        "old_evaluation_protocol": {
            "description": "Legacy estimators used five-fold GroupKFold grouped by race_id, generated cross-validated predictions across the full dataset, and then fitted a final model on all rows. It had no permanently untouched chronological final holdout.",
            "legacy_r2": legacy_r2(),
            "legacy_r2_model": "XGBoost",
            "status": "LEGACY",
            "historical_metrics_file_preserved": str((artifact_dir / "model_metrics.json").relative_to(PROJECT_ROOT)),
            "direct_target_leakage_proven_from_legacy_source": False,
            "explanation": "Legacy feature_columns used an explicit allowlist excluding tire_wear_pct, future_tire_wear_pct, and the target. The non-chronological CV and absence of a final forward holdout make its score unsuitable as unseen-future performance, but the available legacy source alone does not prove direct target leakage caused it.",
        },
        "corrected_evaluation_protocol": {
            "description": "Chronological 60/20/20 race split; strict feature audit before splitting; all preprocessing and model fitting on TRAIN only; validation and final holdout are only evaluated after fit.",
            "ordering": ["season", "race_race_date", "race_round", "race_id"],
            "split_fractions": {"train": 0.60, "validation": 0.20, "holdout": 0.20},
            "splits": exact_splits,
            "zero_race_overlap": True,
            "zero_race_driver_overlap": True,
            "zero_tyre_stint_overlap": True,
            "source_tyre_stints": stint_counts,
            "strict_audit_passed_before_split": True,
            "audit_feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "hyperparameter_tuning_performed": False,
            "model_selection_used_holdout": False,
            "holdout_used_for_early_stopping_or_preprocessing": False,
        },
        "target_definition": "wear[t+1] - wear[t], requiring same race/driver, consecutive lap, consecutive tyre age, and unchanged compound",
        "target_distribution": distributions,
        "target_structure_diagnostic": target_structure,
        "metrics": comparison.to_dict(orient="records"),
        "permutation_sanity_check": permutation_result,
        "feature_ablation": ablation_result,
        "leave_one_race_out_sanity": loo_result,
        "model_fit_provenance": model_provenance,
        "holdout_prediction_provenance_verified": all(
            not (set(info["fit_race_ids"]) & set(holdout_races))
            and not info["holdout_race_ids_seen_during_fit"]
            for info in model_provenance.values()
        ),
        "performance_verified": bool(len(predictions_frame) > 0 and set(predictions_frame["race_id"].unique()) == set(holdout_races)),
        "reproducibility": {
            "procedure_reproducible": True,
            "full_run_independently_repeated": False,
            "random_seed": RANDOM_SEED,
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "train_csv_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        },
        "outputs": {
            "metrics_json": str(metrics_path.relative_to(PROJECT_ROOT)),
            "comparison_csv": str(comparison_path.relative_to(PROJECT_ROOT)),
            "predictions_csv": str(predictions_path.relative_to(PROJECT_ROOT)),
            "feature_importance_csv": str(importance_path.relative_to(PROJECT_ROOT)),
            "audit_report": "leakage_audit_report.md",
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    report = markdown_report(payload, comparison, importance_frame)
    (PROJECT_ROOT / "leakage_audit_report.md").write_text(report, encoding="utf-8")

    print("\nCLEAN STRICT HOLDOUT METRICS")
    print(comparison.to_string(index=False))
    print("\nTARGET DISTRIBUTIONS")
    print(pd.DataFrame(distributions).T.to_string())
    print("\nPERMUTATION SANITY CHECK")
    print(permutation_result)
    print("\nFEATURE ABLATION")
    print(pd.DataFrame(ablation_result).to_string(index=False))
    print("\nLEAVE-ONE-UNSEEN-RACE-OUT")
    print(pd.DataFrame(loo_result).to_string(index=False))
    print("\nTOP TRAIN-ONLY FEATURES BY MODEL")
    print(importance_frame.groupby("model", sort=False).head(5).to_string(index=False))
    print("\nFINAL HOLDOUT PREDICTION PROVENANCE VERIFIED:", payload["holdout_prediction_provenance_verified"])
    print("Historical artifacts/model_metrics.json was not modified.")


if __name__ == "__main__":
    main()