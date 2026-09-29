import os

import joblib
import numpy as np
import pandas as pd


MODEL_FILE = "f1_tyre_wear_model.joblib"
MISSING_VALUE = -999.0

INPUT_FACTORS = [
    ("fuel_load_kg_lag1", "Fuel load one lap ago (kg)"),
    ("lap_time_sec_roll5_std", "Lap-time rolling standard deviation (sec)"),
    ("position_lag1", "Position one lap ago"),
    ("fuel_load_kg_lag2", "Fuel load two laps ago (kg)"),
    ("circuit_base_lap_time_sec", "Circuit base lap time (sec)"),
]


def ask_number(label):
    while True:
        answer = input(f"{label}: ").strip()
        try:
            value = float(answer)
        except ValueError:
            print("Please enter a numeric value.")
            continue

        if not np.isfinite(value):
            print("Please enter a finite numeric value.")
            continue
        return value


def main():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Could not find {MODEL_FILE}. Run train_model.py first."
        )

    artifact = joblib.load(MODEL_FILE)
    feature_names = artifact["feature_columns"]

    print("F1 TYRE DEGRADATION PREDICTOR")
    print("Enter five current/history factors.\n")

    values = {
        feature: ask_number(label)
        for feature, label in INPUT_FACTORS
    }

    row = pd.DataFrame(
        [
            {
                feature: values.get(feature, MISSING_VALUE)
                for feature in feature_names
            }
        ],
        columns=feature_names,
    ).astype(np.float32)

    prediction = float(artifact["model"].predict(row.to_numpy())[0])
    metrics = artifact.get("oof_metrics", {})

    print("\nPREDICTION")
    print(f"Predicted next-lap degradation: {prediction:.6f}%")

    print("\nValidated model metrics")
    print(f"R²   : {metrics.get('delta_r2', float('nan')):.6f}")
    print(f"MAE  : {metrics.get('delta_mae', float('nan')):.6f}")
    print(f"RMSE : {metrics.get('delta_rmse', float('nan')):.6f}")

    print(
        "\nNote: Only five factors were supplied. The remaining model inputs "
        "used the missing-value sentinel, so full CSV predictions are more reliable."
    )


if __name__ == "__main__":
    main()
