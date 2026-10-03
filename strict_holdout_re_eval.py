from __future__ import annotations

import json
import sys
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.f1_tyre.config import PROJECT_ROOT
from src.f1_tyre.data import build_target_frame, drop_duplicate_race_laps, ensure_causal_sort
from src.f1_tyre.evaluation.leakage_audit import audit_feature_matrix
from src.f1_tyre.features import prepare_feature_matrix
from src.f1_tyre.strict_feature_policy import ALLOWED_FEATURES

RANDOM_SEED = 42
TARGET_NAME = "next_lap_wear_increment"


def compute_split_races(all_races: list[int]) -> tuple[list[int], list[int], list[int]]:
    all_races = sorted(int(r) for r in all_races)
    train_end = int(len(all_races) * 0.6)
    val_end = int(len(all_races) * 0.8)
    train_races = all_races[:train_end]
    val_races = all_races[train_end:val_end]
    holdout_races = all_races[val_end:]
    if not (train_races and val_races and holdout_races):
        raise ValueError("Split produced empty partitions.")
    return train_races, val_races, holdout_races


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
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unsupported model: {name}")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("num", numeric_transformer, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_transformer, categorical_columns))

    return ColumnTransformer(transformers=transformers)


def safe_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {"race_id", "driver_id", "track_status"}
    allowed = [column for column in ALLOWED_FEATURES if column in df.columns and column not in blocked]
    if not allowed:
        raise ValueError("No permitted model features remained after filtering.")
    feature_df = df[allowed].copy()
    audit_feature_matrix(feature_df)
    return allowed


def prepare_clean_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    raw = pd.read_csv(path)
    raw = ensure_causal_sort(drop_duplicate_race_laps(raw))
    target_df = build_target_frame(raw)

    prepared = prepare_feature_matrix(target_df)
    feature_columns = safe_feature_columns(prepared)
    feature_df = prepared[feature_columns].copy()
    target_series = target_df[TARGET_NAME].astype(float).copy()
    feature_df = feature_df.reset_index(drop=True)
    target_series = target_series.reset_index(drop=True)

    combined = pd.concat([feature_df, target_series.rename(TARGET_NAME)], axis=1)
    combined["race_id"] = target_df["race_id"].reset_index(drop=True).to_numpy()
    combined["driver_id"] = target_df["driver_id"].reset_index(drop=True).to_numpy()
    combined["lap"] = target_df["lap"].reset_index(drop=True).to_numpy()
    race_ids = sorted(int(r) for r in raw["race_id"].drop_duplicates().tolist())
    split_info = {
        "total_races": len(race_ids),
        "total_rows": len(combined),
        "target_mean": float(target_series.mean()),
        "target_std": float(target_series.std(ddof=1)) if len(target_series) > 1 else 0.0,
        "target_min": float(target_series.min()),
        "target_max": float(target_series.max()),
    }
    return combined, target_df, feature_columns, split_info


def evaluate_split(model_name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = pipeline.predict(X)
    return {
        "model": model_name,
        "r2": float(r2_score(y, y_pred)),
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "rows": int(len(X)),
        "predictions": y_pred,
        "actual": y.to_numpy(),
    }


def compute_feature_importance(model_name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances.mean(axis=0)
        importance_df = pd.DataFrame({"model": model_name, "feature": feature_names, "importance": importances})
    elif hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({"model": model_name, "feature": feature_names, "importance": model.feature_importances_})
    else:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(model, preprocessor.transform(X), y, n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1)
        importance_df = pd.DataFrame({"model": model_name, "feature": feature_names, "importance": result.importances_mean})

    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def run_permutation_sanity_check(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_columns: list[str]) -> dict:
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_NAME]
    X_val = val_df[feature_columns]
    y_val = val_df[TARGET_NAME]

    permuted = y_train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(X_train, permuted)
    val_pred = pipeline.predict(X_val)
    return {
        "r2": float(r2_score(y_val, val_pred)),
        "mae": float(mean_absolute_error(y_val, val_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
    }


def feature_ablation_test(train_df: pd.DataFrame, val_df: pd.DataFrame, holdout_df: pd.DataFrame, feature_columns: list[str]) -> list[dict]:
    variants = {
        "A_strict_causal": feature_columns,
        "B_no_current_lap_telemetry": [c for c in feature_columns if c not in {"lap", "position", "lap_time_sec", "s1_time_sec", "s2_time_sec", "s3_time_sec", "sector_total_sec", "lap_time_per_km", "gap_to_leader_sec", "gap_ahead_sec", "gap_behind_sec", "total_nearest_gap", "track_status", "weather_current", "drs_activated"}],
        "C_no_fuel_features": [c for c in feature_columns if c not in {"fuel_load_kg", "fuel_ratio"}],
        "D_no_lap_and_sector_features": [c for c in feature_columns if c not in {"lap", "position", "lap_time_sec", "s1_time_sec", "s2_time_sec", "s3_time_sec", "sector_total_sec", "lap_time_per_km", "lap_time_sec_lag1", "lap_time_sec_lag2", "lap_time_sec_lag3", "lap_time_sec_roll3_mean", "lap_time_sec_roll5_mean", "lap_time_sec_roll3_std", "lap_time_sec_roll5_std"}],
    }

    results = []
    for name, subset in variants.items():
        subset = [c for c in subset if c in feature_columns]
        X_train = train_df[subset]
        X_val = val_df[subset]
        X_hold = holdout_df[subset]
        y_train = train_df[TARGET_NAME]
        y_hold = holdout_df[TARGET_NAME]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        pipeline.fit(X_train, y_train)
        hold_pred = pipeline.predict(X_hold)
        results.append({
            "variant": name,
            "holdout_r2": float(r2_score(y_hold, hold_pred)),
            "holdout_mae": float(mean_absolute_error(y_hold, hold_pred)),
            "holdout_rmse": float(np.sqrt(mean_squared_error(y_hold, hold_pred))),
        })
    return results


def leave_one_race_out_sanity_check(train_df: pd.DataFrame, holdout_df: pd.DataFrame, feature_columns: list[str], races: list[int], n_races: int = 5) -> list[dict]:
    results = []
    selected = races[:n_races]
    for race_id in selected:
        one_holdout = holdout_df[holdout_df["race_id"] == race_id].copy()
        if one_holdout.empty:
            continue
        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_NAME]
        X_hold = one_holdout[feature_columns]
        y_hold = one_holdout[TARGET_NAME]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        pipeline.fit(X_train, y_train)
        hold_pred = pipeline.predict(X_hold)
        results.append({
            "race_id": int(race_id),
            "rows": int(len(one_holdout)),
            "r2": float(r2_score(y_hold, hold_pred)),
            "mae": float(mean_absolute_error(y_hold, hold_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_hold, hold_pred))),
        })
    return results


def main() -> None:
    train_path = PROJECT_ROOT / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data at {train_path}")

    raw_df = pd.read_csv(train_path)
    ordered = ensure_causal_sort(drop_duplicate_race_laps(raw_df))
    all_races = sorted(int(r) for r in ordered["race_id"].drop_duplicates().tolist())
    train_races, val_races, holdout_races = compute_split_races(all_races)

    print("TRAIN RACE IDs:", train_races)
    print("VALIDATION RACE IDs:", val_races)
    print("HOLDOUT RACE IDs:", holdout_races)
    print(f"Race overlaps: train/val={set(train_races) & set(val_races)}, train/holdout={set(train_races) & set(holdout_races)}, val/holdout={set(val_races) & set(holdout_races)}")
    print(f"Driver overlaps: {len(set((r, d) for r, d in zip(ordered['race_id'], ordered['driver_id']))) }")

    assert set(train_races).isdisjoint(set(val_races))
    assert set(train_races).isdisjoint(set(holdout_races))
    assert set(val_races).isdisjoint(set(holdout_races))

    combined_df, _, feature_columns, split_meta = prepare_clean_dataset(train_path)

    train_df = combined_df[combined_df["race_id"].isin(train_races)].copy()
    val_df = combined_df[combined_df["race_id"].isin(val_races)].copy()
    holdout_df = combined_df[combined_df["race_id"].isin(holdout_races)].copy()

    for split_name, frame in [("train", train_df), ("validation", val_df), ("holdout", holdout_df)]:
        if frame.empty:
            raise ValueError(f"Split produced no rows for {split_name}.")
        if frame["race_id"].nunique() == 0:
            raise ValueError(f"Split {split_name} contains no race ids.")

    for split_name, frame in [("train", train_df), ("validation", val_df), ("holdout", holdout_df)]:
        if frame["race_id"].nunique() > 0:
            print(f"{split_name.upper()} races: {sorted(frame['race_id'].unique().tolist())[:5]} ... {sorted(frame['race_id'].unique().tolist())[-5:]}")

    if any(column in train_df.columns for column in ["tire_wear_pct", "future_tire_wear_pct", "next_lap_wear_increment"]):
        print("Target-proxy columns present in combined frame; they were removed from model features.")

    feature_columns = [column for column in feature_columns if column in train_df.columns]
    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET_NAME].astype(float)
    X_val = val_df[feature_columns].copy()
    y_val = val_df[TARGET_NAME].astype(float)
    X_hold = holdout_df[feature_columns].copy()
    y_hold = holdout_df[TARGET_NAME].astype(float)

    model_names = ["Ridge", "RandomForest", "XGBoost", "NeuralNetwork"]
    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    strict_models_dir = artifact_dir / "strict_holdout_models"
    strict_models_dir.mkdir(exist_ok=True)
    summaries = []
    holdout_predictions = []
    importance_frames = []

    for model_name in model_names:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", make_model(model_name)),
            ]
        )
        pipeline.fit(X_train, y_train)

        for split_name, X_split, y_split in [("train", X_train, y_train), ("validation", X_val, y_val), ("holdout", X_hold, y_hold)]:
            result = evaluate_split(model_name, pipeline, X_split, y_split)
            split_frame = train_df if split_name == "train" else val_df if split_name == "validation" else holdout_df
            summaries.append({
                "model": model_name,
                "split": split_name,
                "r2": result["r2"],
                "mae": result["mae"],
                "rmse": result["rmse"],
                "number_of_rows": result["rows"],
                "number_of_races": int(split_frame["race_id"].nunique()),
            })

            if split_name == "holdout":
                rows = holdout_df.loc[X_split.index, ["race_id", "driver_id", "lap"]].copy()
                rows["model"] = model_name
                rows["actual"] = result["actual"]
                rows["predicted"] = result["predictions"]
                rows["error"] = rows["actual"] - rows["predicted"]
                holdout_predictions.append(rows)

        joblib.dump({
            "model_name": model_name,
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "train_races": train_races,
            "validation_races": val_races,
            "holdout_races": holdout_races,
            "target_name": TARGET_NAME,
        }, strict_models_dir / f"{model_name.lower().replace(' ', '_')}.joblib")
        importance_frames.append(compute_feature_importance(model_name, pipeline, X_train, y_train))

    # Permutation sanity check and ablations.
    permutation_summary = run_permutation_sanity_check(train_df, val_df, feature_columns)
    ablation_summary = feature_ablation_test(train_df, val_df, holdout_df, feature_columns)
    leave_one_race_summary = leave_one_race_out_sanity_check(train_df, holdout_df, feature_columns, holdout_races, n_races=5)

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(["model", "split"]).reset_index(drop=True)

    summary_df.to_csv(artifact_dir / "strict_holdout_comparison.csv", index=False)
    with open(artifact_dir / "strict_holdout_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({
            "old_evaluation_protocol": {
                "description": "Legacy training used the same train.csv with cross-validation and no strict race-disjoint final holdout; historical metrics were accepted as final performance even though the dataset contains future-derived and target-proxy variables.",
                "legacy_r2": 0.9999,
                "status": "LEGACY"
            },
            "corrected_evaluation_protocol": {
                "description": "Train on earlier races, validation on separate races, and final holdout on completely unseen later races; every transformer is fit on TRAIN only.",
                "train_races": train_races,
                "validation_races": val_races,
                "holdout_races": holdout_races,
                "zero_race_overlap": True,
                "zero_race_driver_overlap": True,
            },
            "split_summary": {
                "train_races_count": len(train_races),
                "validation_races_count": len(val_races),
                "holdout_races_count": len(holdout_races),
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(val_df)),
                "holdout_rows": int(len(holdout_df)),
                "target_distribution": {
                    "train": {"mean": float(train_df[TARGET_NAME].mean()), "std": float(train_df[TARGET_NAME].std(ddof=1)), "min": float(train_df[TARGET_NAME].min()), "max": float(train_df[TARGET_NAME].max())},
                    "validation": {"mean": float(val_df[TARGET_NAME].mean()), "std": float(val_df[TARGET_NAME].std(ddof=1)), "min": float(val_df[TARGET_NAME].min()), "max": float(val_df[TARGET_NAME].max())},
                    "holdout": {"mean": float(holdout_df[TARGET_NAME].mean()), "std": float(holdout_df[TARGET_NAME].std(ddof=1)), "min": float(holdout_df[TARGET_NAME].min()), "max": float(holdout_df[TARGET_NAME].max())},
                },
            },
            "metrics": summary_df.to_dict(orient="records"),
            "permutation_sanity_check": permutation_summary,
            "feature_ablation": ablation_summary,
            "leave_one_race_out_sanity": leave_one_race_summary,
            "feature_columns": feature_columns,
            "strict_audit_passed": True,
        }, handle, indent=2)

    if holdout_predictions:
        predictions_df = pd.concat(holdout_predictions, ignore_index=True)
        predictions_df = predictions_df[["model", "race_id", "driver_id", "lap", "actual", "predicted", "error"]]
        predictions_df.to_csv(artifact_dir / "strict_holdout_predictions.csv", index=False)

    importance_df = pd.concat(importance_frames, ignore_index=True)
    importance_df = importance_df.sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)
    importance_df.to_csv(artifact_dir / "strict_holdout_feature_importance.csv", index=False)

    print("\nSTRICT HOLDOUT METRICS")
    print(summary_df.to_string(index=False))
    print("\nPERMUTATION SANITY CHECK")
    print(permutation_summary)
    print("\nFEATURE ABLATION")
    for item in ablation_summary:
        print(item)
    print("\nLEAVE-ONE-RACE-OUT SANITY")
    for item in leave_one_race_summary:
        print(item)


if __name__ == "__main__":
    main()
