import argparse
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_INPUT = "test.csv"
DEFAULT_OUTPUT = "predictions.csv"
DEFAULT_MODEL = "f1_tyre_wear_model.joblib"
GROUP_COLS = ["race_id", "driver_id"]
TARGET_COLUMN = "tire_wear_pct"

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

LAG_SOURCES = [
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
ROLLING_SOURCES = ["lap_time_sec", "s1_time_sec", "s2_time_sec", "s3_time_sec"]

CANDIDATE_STATIC_FEATURES = [
    "season", "race_round", "circuit_id", "circuit_country",
    "circuit_length_km", "circuit_turns", "circuit_drs_zones",
    "circuit_overtake_difficulty", "circuit_base_lap_time_sec",
    "driver_id", "driver_nationality", "driver_skill_rating",
    "driver_aggression_rating", "driver_consistency_rating", "team_id",
    "team_budget_million", "team_car_speed_rating",
    "team_car_downforce_rating", "team_car_reliability_rating",
    "team_pit_crew_rating", "weather", "air_temp_c", "track_temp_c",
    "humidity_pct", "wind_speed_kph", "track_grip_level",
    "safety_car_probability", "race_total_laps", "tyre_change_lap",
    "grid_position", "qualifying_time_sec", "q1_time_sec", "q2_time_sec",
    "q3_time_sec", "lap", "tire_compound", "tire_age_laps",
    *LAG_SOURCES, *[f"{c}_lag{i}" for c in LAG_SOURCES for i in (1, 2, 3)],
    *[f"{c}_roll{w}_mean" for c in ROLLING_SOURCES for w in (3, 5)],
    *[f"{c}_roll5_std" for c in ROLLING_SOURCES],
    "pace_change_1", "pace_change_3", "tire_age_squared", "tire_age_ratio",
]


def sanitize_columns(columns):
    names = []
    counts = {}
    for column in columns:
        base = re.sub(r"[\[\]<>]", "_", str(column))
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        names.append(base if occurrence == 0 else f"{base}_{occurrence}")
    return names


def add_group_lags(df):
    for column in LAG_SOURCES:
        if column not in df.columns:
            continue
        grouped = df.groupby(GROUP_COLS, sort=False)[column]
        for lag in (1, 2, 3):
            df[f"{column}_lag{lag}"] = grouped.shift(lag)
    return df


def add_shifted_rolling(df):
    for column in ROLLING_SOURCES:
        if column not in df.columns:
            continue
        shifted = df.groupby(GROUP_COLS, sort=False)[column].shift(1)
        grouped = shifted.groupby(
            [df[group] for group in GROUP_COLS], sort=False
        )
        for window in (3, 5):
            df[f"{column}_roll{window}_mean"] = (
                grouped.rolling(window=window, min_periods=2)
                .mean().reset_index(level=GROUP_COLS, drop=True)
            )
            if window == 5:
                df[f"{column}_roll5_std"] = (
                    grouped.rolling(window=window, min_periods=3)
                    .std().reset_index(level=GROUP_COLS, drop=True)
                )
    return df


def add_features(df):
    required = ["race_id", "driver_id", "lap"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))

    df = df.copy()
    df["_input_order"] = np.arange(len(df))
    sort_columns = ["race_id", "driver_id", "lap"]
    df = df.sort_values(sort_columns).reset_index(drop=True)

    df = add_group_lags(df)
    df = add_shifted_rolling(df)

    if "tire_age_laps" in df.columns:
        age = pd.to_numeric(df["tire_age_laps"], errors="coerce")
        df["tire_age_squared"] = age ** 2

    if "race_total_laps" in df.columns and "tire_age_laps" in df.columns:
        denominator = pd.to_numeric(
            df["race_total_laps"], errors="coerce"
        ).replace(0, np.nan)
        df["tire_age_ratio"] = (
            pd.to_numeric(df["tire_age_laps"], errors="coerce") / denominator
        )

    return df


def build_matrix(df, artifact):
    feature_names = artifact["feature_columns"]
    categorical_columns = artifact["categorical_columns"]

    raw_columns = [
        column for column in CANDIDATE_STATIC_FEATURES
        if column in df.columns and column not in DETERMINISTIC_TARGET_FEATURES
    ]
    matrix = df[raw_columns].copy()

    active_categories = [
        column for column in categorical_columns if column in matrix.columns
    ]
    for column in active_categories:
        matrix[column] = matrix[column].astype("string").fillna("UNKNOWN")

    matrix = pd.get_dummies(
        matrix,
        columns=active_categories,
        dummy_na=True,
    )
    matrix.columns = sanitize_columns(matrix.columns)

    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")

    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    matrix = matrix.reindex(columns=feature_names, fill_value=-999.0)
    return matrix.astype(np.float32)


def calculate_metrics(df, predictions):
    if TARGET_COLUMN not in df.columns:
        return None

    evaluation = df.copy()
    evaluation["_prediction"] = predictions
    evaluation["_next_wear"] = (
        evaluation.groupby(GROUP_COLS, sort=False)[TARGET_COLUMN].shift(-1)
    )
    evaluation["_next_lap"] = (
        evaluation.groupby(GROUP_COLS, sort=False)["lap"].shift(-1)
    )
    evaluation["_next_compound"] = (
        evaluation.groupby(GROUP_COLS, sort=False)["tire_compound"].shift(-1)
        if "tire_compound" in evaluation.columns
        else np.nan
    )

    valid = (
        evaluation["_next_lap"].eq(evaluation["lap"] + 1)
        & evaluation["_next_wear"].notna()
        & evaluation["_next_compound"].astype("string").eq(
            evaluation["tire_compound"].astype("string")
        )
    )
    if not valid.any():
        return None

    actual = (
        evaluation.loc[valid, "_next_wear"]
        - evaluation.loc[valid, TARGET_COLUMN]
    ).to_numpy(dtype=float)
    predicted = evaluation.loc[valid, "_prediction"].to_numpy(dtype=float)
    return {
        "r2": r2_score(actual, predicted),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "rows": int(valid.sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict next-lap tyre wear degradation."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    artifact = joblib.load(args.model)
    source = pd.read_csv(args.input)
    featured = add_features(source)
    matrix = build_matrix(featured, artifact)
    predictions = artifact["model"].predict(matrix.to_numpy())

    output = featured[["race_id", "driver_id", "lap"]].copy()
    output["predicted_degradation"] = predictions
    if TARGET_COLUMN in featured.columns:
        output["current_wear_pct"] = featured[TARGET_COLUMN].to_numpy()
        output["predicted_next_wear_pct"] = (
            output["current_wear_pct"] + predictions
        ).clip(0, 100)
    output = output.sort_values("lap").reset_index(drop=True)
    output.to_csv(args.output, index=False)

    print("Prediction complete.")
    print(f"Rows predicted: {len(output):,}")
    print(f"Predictions saved: {args.output}")
    print("\nValidated Δwear metrics from training:")
    metrics = artifact.get("oof_metrics", {})
    print(f"  R²   : {metrics.get('delta_r2', float('nan')):.6f}")
    print(f"  MAE  : {metrics.get('delta_mae', float('nan')):.6f}")
    print(f"  RMSE : {metrics.get('delta_rmse', float('nan')):.6f}")

    input_metrics = calculate_metrics(featured, predictions)
    if input_metrics is not None:
        print("\nMetrics calculated from this input:")
        print(f"  R²   : {input_metrics['r2']:.6f}")
        print(f"  MAE  : {input_metrics['mae']:.6f}")
        print(f"  RMSE : {input_metrics['rmse']:.6f}")
        print(f"  Rows : {input_metrics['rows']:,}")
    else:
        print(
            "\nInput metrics unavailable: input must contain true sequential "
            "next-lap wear values."
        )


if __name__ == "__main__":
    main()
