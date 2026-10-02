from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from f1_tyre.model_loader import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interactive inference with an existing model.")
    parser.add_argument("--model", default="xgboost", choices=["ridge", "random_forest", "xgboost", "neural_network"])
    args = parser.parse_args()
    artifact = load_model(args.model)
    features = artifact.get("feature_columns", [])
    print(f"Loaded model: {args.model}")
    print("Model metadata:", artifact.get("metadata", {}))
    print("Metrics:", artifact.get("metrics", {}))
    current_wear = float(input("Current tyre wear (%): "))

    row = {}
    for feature in features:
        value = input(f"{feature}: ")
        if value.strip() in {"", "None", "nan"}:
            raise ValueError(f"Missing value for required feature '{feature}'.")
        try:
            row[feature] = float(value)
        except ValueError:
            row[feature] = value

    frame = pd.DataFrame([row])
    prediction = artifact["model"].predict(frame[features])[0]
    next_wear = current_wear + float(prediction)
    print(f"Predicted delta wear: {float(prediction):.6f}")
    print(f"Predicted next tyre wear: {next_wear:.6f}")


if __name__ == "__main__":
    main()
