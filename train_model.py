import os
import re
import warnings
from multiprocessing import cpu_count

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score, 
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_FILE = "train.csv"

MODEL_FILE = "f1_tyre_wear_model.joblib"
IMPORTANCE_FILE = "feature_importance.csv"
OOF_FILE = "oof_predictions.csv"

RANDOM_STATE = 42
N_SPLITS = 5

GROUP_COLS = ["race_id", "driver_id"]

TARGET_COLUMN = "tire_wear_pct"

# Use GPU when available.
N_JOBS = max(1, cpu_count() - 2)


# ============================================================================
# XGBOOST HARDWARE CHECK
# ============================================================================

def verify_gpu():
    print("\n" + "=" * 75)
    print("XGBOOST HARDWARE")
    print("=" * 75)

    print("XGBoost version :", __import__("xgboost").__version__)
    print("CPU threads     :", cpu_count())

    try:
        test_model = XGBRegressor(
            n_estimators=2,
            max_depth=2,
            learning_rate=0.1,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )

        X_test = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y_test = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        test_model.fit(X_test, y_test)

        print("GPU successfully verified.")
        return "cuda"

    except Exception as exc:
        print("GPU unavailable. Falling back to CPU.")
        print("Reason:", str(exc))
        return "cpu"


DEVICE = verify_gpu()


# ============================================================================
# HELPERS
# ============================================================================

def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_group_lags(df, group_cols, source_columns, lags=(1, 2, 3)):
    created = []

    for col in source_columns:
        if col not in df.columns:
            continue

        grouped = df.groupby(group_cols, sort=False)[col]

        for lag in lags:
            new_col = f"{col}_lag{lag}"
            df[new_col] = grouped.shift(lag)
            created.append(new_col)

    return df, created


def add_shifted_rolling_features(
    df,
    group_cols,
    source_columns,
    windows=(3, 5),
):
    created = []

    for col in source_columns:
        if col not in df.columns:
            continue

        shifted = df.groupby(group_cols, sort=False)[col].shift(1)

        for window in windows:
            mean_col = f"{col}_roll{window}_mean"
            std_col = f"{col}_roll{window}_std"

            df[mean_col] = (
                shifted.groupby(
                    [df[g] for g in group_cols],
                    sort=False,
                )
                .rolling(window=window, min_periods=2)
                .mean()
                .reset_index(level=group_cols, drop=True)
            )

            if window >= 5:
                df[std_col] = (
                    shifted.groupby(
                        [df[g] for g in group_cols],
                        sort=False,
                    )
                    .rolling(window=window, min_periods=3)
                    .std()
                    .reset_index(level=group_cols, drop=True)
                )
                created.append(std_col)

            created.append(mean_col)

    return df, created


# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 75)
print("F1 STRICT-CAUSAL TYRE WEAR MODEL")
print("=" * 75)

print("\n" + "=" * 75)
print("LOADING DATA")
print("=" * 75)

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Could not find '{TRAIN_FILE}'. "
        f"Put train.csv in the same folder as this script."
    )

df = pd.read_csv(TRAIN_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns    : {len(df.columns)}")

require_columns(
    df,
    [
        "race_id",
        "driver_id",
        "lap",
        "tire_compound",
        "tire_age_laps",
        TARGET_COLUMN,
    ],
)

# Sort first so all temporal operations are causal.
sort_cols = [c for c in ["race_id", "driver_id", "lap"] if c in df.columns]

print("\nSorting race data...")
df = df.sort_values(sort_cols).reset_index(drop=True)


# ============================================================================
# DUPLICATE CHECK
# ============================================================================

duplicate_key = ["race_id", "driver_id", "lap"]
duplicate_count = int(df.duplicated(duplicate_key).sum())

print(f"Duplicate race/driver/lap rows: {duplicate_count:,}")

if duplicate_count > 0:
    print("Removing duplicate race/driver/lap rows...")
    df = (
        df.drop_duplicates(duplicate_key, keep="first")
          .sort_values(sort_cols)
          .reset_index(drop=True)
    )


# ============================================================================
# CORE DATA CLEANING
# ============================================================================

core_numeric = ["race_id", "driver_id", "lap", TARGET_COLUMN, "tire_age_laps"]
df = safe_numeric(df, core_numeric)

before = len(df)

df = df.dropna(
    subset=[
        "race_id",
        "driver_id",
        "lap",
        TARGET_COLUMN,
        "tire_age_laps",
        "tire_compound",
    ]
).copy()

removed = before - len(df)

print(f"Rows removed because of missing core data: {removed:,}")


# ============================================================================
# TARGET SANITY
# ============================================================================

print("\n" + "=" * 75)
print("TARGET SANITY CHECK")
print("=" * 75)

print(f"Target min : {df[TARGET_COLUMN].min():.6f}")
print(f"Target max : {df[TARGET_COLUMN].max():.6f}")
print(f"Target mean: {df[TARGET_COLUMN].mean():.6f}")
print(f"Target std : {df[TARGET_COLUMN].std():.6f}")


# ============================================================================
# LEAKAGE AUDIT
# ============================================================================

print("\n" + "=" * 75)
print("LEAKAGE AUDIT")
print("=" * 75)

# These columns are not allowed into the feature matrix.
# IMPORTANT: tire_wear_pct is intentionally NOT dropped from df yet.
# It is required to calculate the target:
#     next_wear - current_wear
BLOCKED_FEATURE_COLUMNS = {
    # Race outcome / post-race information
    "finish_position",
    "finishing_position",
    "position_finish",
    "status",
    "points",

    # Fastest-lap information can depend on later laps
    "fastest_lap",
    "fastest_lap_time",
    "fastest_lap_speed",

    # Pit-stop outcomes / future tyre state
    "pit_stop_this_lap",
    "pit_stop_duration_sec",
    "pit_new_compound",
    "pit_lap",

    # End-of-race weather
    "race_weather_end",

    # Any explicitly future fields
    "next_lap_time",
    "future_lap_time",
    "future_tire_wear",
}

for col in sorted(BLOCKED_FEATURE_COLUMNS):
    if col in df.columns:
        print(f"  [BLOCKED] {col}")

print(f"  [TARGET - KEPT TEMPORARILY] {TARGET_COLUMN}")


# ============================================================================
# CAUSAL TELEMETRY
# ============================================================================

print("\n" + "=" * 75)
print("CREATING CAUSAL TELEMETRY FEATURES")
print("=" * 75)

# Current-lap telemetry is NOT directly used as a predictive feature.
# Instead, only information from completed previous laps is used.
lag_sources = [
    "lap_time_sec",
    "s1_time_sec",
    "s2_time_sec",
    "s3_time_sec",
    "position",
    "gap_to_leader_sec",
    "gap_ahead_sec",
    "gap_behind_sec",
    "fuel_load_kg",
    "ers_deploy_pct",
    "ers_harvest_pct",
    "race_track_temp_c",
    "race_air_temp_c",
    "race_humidity_pct",
    "wind_speed_kph",
    "track_grip_level",
]

df, lagged_columns = add_group_lags(
    df,
    GROUP_COLS,
    lag_sources,
    lags=(1, 2, 3),
)

print(f"Lagged features created: {len(lagged_columns)}")


# ============================================================================
# SHIFTED ROLLING TELEMETRY
# ============================================================================

print("\nCreating historical rolling telemetry...")

rolling_sources = [
    "lap_time_sec",
    "s1_time_sec",
    "s2_time_sec",
    "s3_time_sec",
]

# Rolling calculations are based on shifted data, so the current lap
# cannot enter its own rolling history.
df, rolling_columns = add_shifted_rolling_features(
    df,
    GROUP_COLS,
    rolling_sources,
    windows=(3, 5),
)

print(f"Rolling features created: {len(rolling_columns)}")


# ============================================================================
# PACE TREND FEATURES
# ============================================================================

print("\nCreating historical pace trends...")

if "lap_time_sec" in df.columns:
    lap_time_lag1 = (
        df.groupby(GROUP_COLS, sort=False)["lap_time_sec"]
          .shift(1)
    )

    lap_time_lag3 = (
        df.groupby(GROUP_COLS, sort=False)["lap_time_sec"]
          .shift(3)
    )

    df["pace_change_1"] = (
        df["lap_time_sec_lag1"] - lap_time_lag1
    )

    df["pace_change_3"] = (
        df["lap_time_sec_lag1"] - lap_time_lag3
    )

    # The above pace_change_1 is deliberately rewritten below to ensure
    # it uses only completed previous laps:
    df["pace_change_1"] = (
        df["lap_time_sec_lag1"] -
        df.groupby(GROUP_COLS, sort=False)["lap_time_sec"].shift(2)
    )

    df["pace_change_3"] = (
        df["lap_time_sec_lag1"] -
        df.groupby(GROUP_COLS, sort=False)["lap_time_sec"].shift(4)
    )

    print("Pace trend features created: 2")
else:
    print("No lap_time_sec column found; pace trends skipped.")


# ============================================================================
# TYRE FEATURES
# ============================================================================

print("\nCreating tyre features...")

# These are known at the current lap before predicting the next lap.
df["tire_age_squared"] = (
    pd.to_numeric(df["tire_age_laps"], errors="coerce") ** 2
)

# Optional normalized age. Avoid divide-by-zero.
if "race_total_laps" in df.columns:
    denom = pd.to_numeric(df["race_total_laps"], errors="coerce").replace(0, np.nan)
    df["tire_age_ratio"] = (
        pd.to_numeric(df["tire_age_laps"], errors="coerce") / denom
    )


# ============================================================================
# NEXT-LAP TARGET
# ============================================================================

print("\nCreating true next-lap wear-increment target...")

# This is the critical fix:
# tire_wear_pct remains in df so we can build y.
df["_current_wear"] = df[TARGET_COLUMN]

df["_next_wear"] = (
    df.groupby(GROUP_COLS, sort=False)[TARGET_COLUMN]
      .shift(-1)
)

df["_next_lap"] = (
    df.groupby(GROUP_COLS, sort=False)["lap"]
      .shift(-1)
)

if "tire_compound" in df.columns:
    df["_next_compound"] = (
        df.groupby(GROUP_COLS, sort=False)["tire_compound"]
          .shift(-1)
    )
else:
    df["_next_compound"] = np.nan

# Require:
#   1. The following row is truly the immediate next lap.
#   2. The tyre compound does not change.
#   3. The next wear value exists.
#
# This prevents crossing pit stops / compound transitions.
valid_next_lap = (
    df["_next_lap"] == df["lap"] + 1
)

valid_same_compound = (
    df["_next_compound"].astype("string") ==
    df["tire_compound"].astype("string")
)

valid_target = df["_next_wear"].notna()

valid_rows = (
    valid_next_lap &
    valid_same_compound &
    valid_target
)

before_target_filter = len(df)

df = df[valid_rows].copy()

print(
    f"Rows removed at next-lap/compound boundary: "
    f"{before_target_filter - len(df):,}"
)

# True degradation target:
# How much additional tyre wear occurs during the next lap?
df["target"] = (
    df["_next_wear"] - df["_current_wear"]
)

print(f"Rows available for training: {len(df):,}")

print("\nWear increment sanity:")
print(f"  Mean Δwear : {df['target'].mean():.6f}")
print(f"  Std  Δwear : {df['target'].std():.6f}")
print(f"  Min  Δwear : {df['target'].min():.6f}")
print(f"  Max  Δwear : {df['target'].max():.6f}")


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 75)
print("BUILDING FINAL FEATURE MATRIX")
print("=" * 75)

# These can be used because they describe information known before the
# next lap starts and are not future race outcomes.
candidate_static_features = [
    "season",
    "race_round",

    "circuit_id",
    "circuit_country",
    "circuit_length_km",
    "circuit_turns",
    "circuit_drs_zones",
    "circuit_overtake_difficulty",
    "circuit_base_lap_time_sec",

    "driver_id",
    "driver_nationality",
    "driver_skill_rating",
    "driver_aggression_rating",
    "driver_consistency_rating",

    "team_id",
    "team_budget_million",
    "team_car_speed_rating",
    "team_car_downforce_rating",
    "team_car_reliability_rating",
    "team_pit_crew_rating",

    "weather",
    "air_temp_c",
    "track_temp_c",
    "humidity_pct",
    "wind_speed_kph",
    "track_grip_level",
    "safety_car_probability",

    "race_total_laps",
    "tyre_change_lap",

    "grid_position",
    "qualifying_time_sec",
    "q1_time_sec",
    "q2_time_sec",
    "q3_time_sec",

    # Current information known at the prediction point
    "lap",
    "tire_compound",
    "tire_age_laps",

    # Causal lagged telemetry
    *lagged_columns,
    *rolling_columns,

    "pace_change_1",
    "pace_change_3",

    "tire_age_squared",
    "tire_age_ratio",
]

# These fields directly encode the synthetic wear progression in this data.
# Exclude them before one-hot encoding so derived columns cannot re-enter X.
DETERMINISTIC_TARGET_FEATURES = {
    "lap",
    "tire_compound",
    "tire_age_laps",
    "tire_age_squared",
    "tire_age_ratio",
    "tyre_change_lap",
    "pace_change_1",
    "pace_change_3",
}

# Only retain fields actually present.
feature_columns = [
    c for c in candidate_static_features
    if c in df.columns
]

# Explicitly remove anything that can leak.
FINAL_EXCLUDE = (
    set(BLOCKED_FEATURE_COLUMNS)
    | {
        TARGET_COLUMN,      # current tyre wear is NOT an input
        "target",
        "_current_wear",
        "_next_wear",
        "_next_lap",
        "_next_compound",
    }
)

feature_columns = [
    c for c in feature_columns
    if c not in FINAL_EXCLUDE
    and c not in DETERMINISTIC_TARGET_FEATURES
]

X = df[feature_columns].copy()
y = df["target"].copy()

groups = df["race_id"].copy()


# ============================================================================
# FINAL LEAKAGE CHECK
# ============================================================================

print("\n" + "=" * 75)
print("FINAL LEAKAGE CHECK")
print("=" * 75)

leakage_patterns = (
    "target",
    "future",
    "next_",
    "finish",
    "points",
    "fastest",
    "pit_stop",
    "pit_new",
)

bad_features = [
    c for c in X.columns
    if any(pattern in c.lower() for pattern in leakage_patterns)
]

if bad_features:
    raise RuntimeError(
        "Potential leakage detected in final features:\n"
        + "\n".join(f"  - {c}" for c in bad_features)
    )

if TARGET_COLUMN in X.columns:
    raise RuntimeError(
        f"Direct target leakage detected: '{TARGET_COLUMN}' is in X."
    )

if "_current_wear" in X.columns:
    raise RuntimeError(
        "Current tyre wear accidentally entered X. "
        "It must only be used to reconstruct next-lap wear."
    )

if len(X) != len(y):
    raise RuntimeError(
        f"Feature/target length mismatch: X={len(X)}, y={len(y)}"
    )

print("Leakage check passed.")
print(
    "Deterministic target features excluded: "
    + ", ".join(
        sorted(
            DETERMINISTIC_TARGET_FEATURES.intersection(
                candidate_static_features
            )
        )
    )
)
print(f"Features before encoding: {X.shape[1]}")
print(f"Rows                         : {len(X):,}")


# ============================================================================
# ONE-HOT ENCODING
# ============================================================================

print("\nEncoding categorical features...")

categorical_columns = [
    c for c in [
        "tire_compound",
        "circuit_country",
        "driver_nationality",
        "weather",
    ]
    if c in X.columns
]

# Convert categoricals to strings, retaining missing values as a category.
for col in categorical_columns:
    X[col] = X[col].astype("string").fillna("UNKNOWN")

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    dummy_na=True,
)

# XGBoost rejects feature names containing square brackets or angle brackets.
# One-hot categories can contain those characters, so sanitize names after
# encoding while keeping every feature name unique.
sanitized_names = []
name_counts = {}

for column in X.columns:
    base_name = re.sub(r"[\[\]<>]", "_", str(column))
    occurrence = name_counts.get(base_name, 0)
    name_counts[base_name] = occurrence + 1
    sanitized_names.append(
        base_name if occurrence == 0 else f"{base_name}_{occurrence}"
    )

X.columns = sanitized_names

# Make all columns numeric.
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Missing telemetry is expected at the beginning of a race/history window.
# Use a sentinel rather than dropping large numbers of otherwise valid rows.
X = X.replace([np.inf, -np.inf], np.nan).fillna(-999.0)

# XGBoost works more efficiently with float32 for large datasets.
X = X.astype(np.float32)

print(f"Features after encoding: {X.shape[1]}")


# ============================================================================
# GROUPED CROSS VALIDATION
# ============================================================================

print("\n" + "=" * 75)
print("GROUPED CROSS-VALIDATION")
print("=" * 75)

unique_races = groups.nunique()

if unique_races < N_SPLITS:
    raise RuntimeError(
        f"Only {unique_races} unique races available, "
        f"but N_SPLITS={N_SPLITS}."
    )

print(f"Unique races: {unique_races}")
print(f"Group column: race_id")
print("No race is allowed to appear in both training and validation.")


# ============================================================================
# XGBOOST PARAMETERS
# ============================================================================

params = dict(
    n_estimators=3000,
    max_depth=7,
    learning_rate=0.025,
    min_child_weight=8,
    gamma=0.05,

    reg_alpha=0.10,
    reg_lambda=2.0,

    subsample=0.85,
    colsample_bytree=0.85,

    objective="reg:squarederror",
    eval_metric="rmse",

    tree_method="hist",
    device=DEVICE,

    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
)

print("\nModel:")
print("  n_estimators       :", params["n_estimators"])
print("  max_depth          :", params["max_depth"])
print("  learning_rate      :", params["learning_rate"])
print("  min_child_weight   :", params["min_child_weight"])
print("  subsample          :", params["subsample"])
print("  colsample_bytree   :", params["colsample_bytree"])
print("  device             :", params["device"])
print("  n_jobs             :", params["n_jobs"])


# ============================================================================
# TRAIN / VALIDATION
# ============================================================================

gkf = GroupKFold(n_splits=N_SPLITS)

oof_delta = np.full(len(df), np.nan, dtype=np.float64)
oof_next_wear = np.full(len(df), np.nan, dtype=np.float64)

fold_results = []
feature_importance_accumulator = np.zeros(X.shape[1], dtype=np.float64)

feature_names = X.columns.to_list()

for fold, (train_idx, valid_idx) in enumerate(
    gkf.split(X, y, groups=groups),
    start=1,
):
    print("\n" + "=" * 75)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 75)

    train_races = set(groups.iloc[train_idx].unique())
    valid_races = set(groups.iloc[valid_idx].unique())

    overlap = train_races.intersection(valid_races)

    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(overlap)} races overlap "
            f"between train and validation."
        )

    print(f"Train rows : {len(train_idx):,}")
    print(f"Valid rows : {len(valid_idx):,}")
    print(f"Train races: {len(train_races):,}")
    print(f"Valid races: {len(valid_races):,}")

    model = XGBRegressor(**params)

    model.fit(
        X.iloc[train_idx].to_numpy(),
        y.iloc[train_idx],
        eval_set=[
            (
                X.iloc[valid_idx].to_numpy(),
                y.iloc[valid_idx],
            )
        ],
        verbose=False,
    )

    pred_delta = model.predict(X.iloc[valid_idx].to_numpy())

    # Reconstruct predicted absolute wear:
    # current observed wear + predicted increase
    current_wear_valid = df["_current_wear"].iloc[valid_idx].to_numpy(
        dtype=np.float64
    )

    actual_next_wear = df["_next_wear"].iloc[valid_idx].to_numpy(
        dtype=np.float64
    )

    pred_next_wear = current_wear_valid + pred_delta

    oof_delta[valid_idx] = pred_delta
    oof_next_wear[valid_idx] = pred_next_wear

    delta_actual = y.iloc[valid_idx].to_numpy(dtype=np.float64)

    delta_r2 = r2_score(delta_actual, pred_delta)
    delta_mae = mean_absolute_error(delta_actual, pred_delta)
    delta_rmse = np.sqrt(
        mean_squared_error(delta_actual, pred_delta)
    )

    next_r2 = r2_score(actual_next_wear, pred_next_wear)
    next_mae = mean_absolute_error(actual_next_wear, pred_next_wear)
    next_rmse = np.sqrt(
        mean_squared_error(actual_next_wear, pred_next_wear)
    )

    fold_results.append(
        {
            "fold": fold,
            "delta_r2": delta_r2,
            "delta_mae": delta_mae,
            "delta_rmse": delta_rmse,
            "next_wear_r2": next_r2,
            "next_wear_mae": next_mae,
            "next_wear_rmse": next_rmse,
        }
    )

    feature_importance_accumulator += model.feature_importances_

    print("\nΔWEAR METRICS")
    print(f"  R²   : {delta_r2:.10f}")
    print(f"  MAE  : {delta_mae:.6f}")
    print(f"  RMSE : {delta_rmse:.6f}")

    print("\nRECONSTRUCTED NEXT-WEAR METRICS")
    print(f"  R²   : {next_r2:.10f}")
    print(f"  MAE  : {next_mae:.6f}")
    print(f"  RMSE : {next_rmse:.6f}")


# ============================================================================
# OOF METRICS
# ============================================================================

valid_oof = np.isfinite(oof_delta) & np.isfinite(oof_next_wear)

actual_delta_oof = y.to_numpy(dtype=np.float64)[valid_oof]
pred_delta_oof = oof_delta[valid_oof]

actual_next_oof = df["_next_wear"].to_numpy(dtype=np.float64)[valid_oof]
pred_next_oof = oof_next_wear[valid_oof]

delta_r2 = r2_score(actual_delta_oof, pred_delta_oof)
delta_mae = mean_absolute_error(actual_delta_oof, pred_delta_oof)
delta_rmse = np.sqrt(
    mean_squared_error(actual_delta_oof, pred_delta_oof)
)

next_r2 = r2_score(actual_next_oof, pred_next_oof)
next_mae = mean_absolute_error(actual_next_oof, pred_next_oof)
next_rmse = np.sqrt(
    mean_squared_error(actual_next_oof, pred_next_oof)
)

fold_df = pd.DataFrame(fold_results)


print("\n" + "=" * 75)
print("FINAL OUT-OF-FOLD PERFORMANCE")
print("=" * 75)

print("\nTRUE ΔWEAR TARGET")
print(f"R²   : {delta_r2:.12f}")
print(f"MAE  : {delta_mae:.8f}")
print(f"RMSE : {delta_rmse:.8f}")

print("\nRECONSTRUCTED NEXT-LAP TYRE WEAR")
print(f"R²   : {next_r2:.12f}")
print(f"MAE  : {next_mae:.8f}")
print(f"RMSE : {next_rmse:.8f}")

print("\nMEAN / STD OF FOLD R²")
print(
    f"Δwear R²       : "
    f"{fold_df['delta_r2'].mean():.10f} ± "
    f"{fold_df['delta_r2'].std():.10f}"
)
print(
    f"Next-wear R²   : "
    f"{fold_df['next_wear_r2'].mean():.10f} ± "
    f"{fold_df['next_wear_r2'].std():.10f}"
)


# ============================================================================
# DIAGNOSTICS
# ============================================================================

print("\n" + "=" * 75)
print("DIAGNOSTICS")
print("=" * 75)

delta_residual = actual_delta_oof - pred_delta_oof
next_residual = actual_next_oof - pred_next_oof

delta_target_std = np.std(actual_delta_oof)
next_target_std = np.std(actual_next_oof)

print(f"Δwear target std       : {delta_target_std:.10f}")
print(f"Δwear residual std     : {np.std(delta_residual):.10f}")
print(f"Δwear max abs error    : {np.max(np.abs(delta_residual)):.10f}")

print(f"Next-wear target std   : {next_target_std:.10f}")
print(f"Next-wear residual std : {np.std(next_residual):.10f}")
print(f"Next-wear max abs err  : {np.max(np.abs(next_residual)):.10f}")

if np.std(actual_delta_oof) > 0 and np.std(pred_delta_oof) > 0:
    delta_corr = np.corrcoef(
        actual_delta_oof,
        pred_delta_oof,
    )[0, 1]
else:
    delta_corr = np.nan

if np.std(actual_next_oof) > 0 and np.std(pred_next_oof) > 0:
    next_corr = np.corrcoef(
        actual_next_oof,
        pred_next_oof,
    )[0, 1]
else:
    next_corr = np.nan

print(f"Δwear actual/pred corr  : {delta_corr:.10f}")
print(f"Next-wear corr          : {next_corr:.10f}")

# A constant baseline for Δwear.
baseline_delta = np.full_like(
    actual_delta_oof,
    np.mean(actual_delta_oof),
)

baseline_r2 = r2_score(actual_delta_oof, baseline_delta)
baseline_mae = mean_absolute_error(
    actual_delta_oof,
    baseline_delta,
)

print("\nΔwear mean baseline")
print(f"R²   : {baseline_r2:.10f}")
print(f"MAE  : {baseline_mae:.8f}")


# ============================================================================
# SANITY CHECK FOR SUSPICIOUSLY HIGH R2
# ============================================================================

if next_r2 > 0.999:
    print(
        "\nWARNING: Reconstructed next-wear R² is extremely high."
    )
    print(
        "This is not automatically leakage because current wear is "
        "used only for reconstruction, not as a model feature."
    )
    print(
        "However, inspect the data generation process if this remains "
        "near-perfect. Synthetic targets can be mathematically encoded "
        "by tyre age, race progression, or another field."
    )

if delta_r2 > 0.999:
    print(
        "\nWARNING: Δwear R² is extremely high."
    )
    print(
        "Inspect whether wear increments are generated from a deterministic "
        "formula involving one or more input columns."
    )


# ============================================================================
# TRAIN FINAL MODEL
# ============================================================================

print("\n" + "=" * 75)
print("TRAINING FINAL MODEL ON ALL RACES")
print("=" * 75)

final_model = XGBRegressor(**params)

final_model.fit(
    X.to_numpy(),
    y,
    verbose=False,
)

print("Final model training complete.")


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": final_model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False,
)

print(f"Feature importance saved: {IMPORTANCE_FILE}")

print("\nTOP 30 FEATURES")
print(importance_df.head(30).to_string(index=False))


# ============================================================================
# SAVE OOF PREDICTIONS
# ============================================================================

oof_output = df[
    [
        "race_id",
        "driver_id",
        "lap",
        "tire_compound",
        "tire_age_laps",
        "_current_wear",
        "_next_wear",
    ]
].copy()

oof_output["actual_delta_wear"] = y.to_numpy()
oof_output["predicted_delta_wear"] = oof_delta
oof_output["delta_error"] = (
    oof_output["actual_delta_wear"] -
    oof_output["predicted_delta_wear"]
)

oof_output["actual_next_wear"] = oof_output["_next_wear"]
oof_output["predicted_next_wear"] = (
    oof_output["_current_wear"] +
    oof_output["predicted_delta_wear"]
)

oof_output["next_wear_error"] = (
    oof_output["actual_next_wear"] -
    oof_output["predicted_next_wear"]
)

oof_output.to_csv(
    OOF_FILE,
    index=False,
)

print(f"\nOOF predictions saved: {OOF_FILE}")


# ============================================================================
# SAVE MODEL ARTIFACT
# ============================================================================

artifact = {
    "model": final_model,
    "feature_columns": feature_names,
    "categorical_columns": categorical_columns,

    # Important metadata
    "target_source": TARGET_COLUMN,
    "prediction_target": "next_lap_wear_increment",
    "prediction_formula": "current_wear + predicted_delta_wear",

    "causal": True,
    "strict_next_lap": True,
    "compound_boundary_filtered": True,

    "group_column": "race_id",
    "group_cols": GROUP_COLS,

    "device": DEVICE,
    "random_state": RANDOM_STATE,

    "oof_metrics": {
        "delta_r2": float(delta_r2),
        "delta_mae": float(delta_mae),
        "delta_rmse": float(delta_rmse),
        "next_wear_r2": float(next_r2),
        "next_wear_mae": float(next_mae),
        "next_wear_rmse": float(next_rmse),
    },
}

joblib.dump(
    artifact,
    MODEL_FILE,
)

print(f"Model saved: {MODEL_FILE}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 75)
print("TRAINING COMPLETE")
print("=" * 75)

print(f"Rows used                 : {len(df):,}")
print(f"Input features            : {X.shape[1]:,}")
print(f"Unique races              : {groups.nunique():,}")
print(f"Δwear OOF R²              : {delta_r2:.12f}")
print(f"Next-wear reconstructed R²: {next_r2:.12f}")
print(f"Device                    : {DEVICE}")

print("\nGenerated files:")
print(f"  {MODEL_FILE}")
print(f"  {IMPORTANCE_FILE}")
print(f"  {OOF_FILE}")

print("\nImportant:")
print("  - tire_wear_pct was never used as an X feature.")
print("  - Only lagged telemetry is used.")
print("  - Future laps are not used.")
print("  - Pit/compound boundaries are excluded.")
print("  - Train/validation is grouped by race_id.")
print("  - The main target is next-lap wear increase (Δwear).")
print("  - Absolute next-lap wear is reconstructed from current observed wear + Δwear.")