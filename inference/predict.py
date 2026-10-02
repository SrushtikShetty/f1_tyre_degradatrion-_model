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
    parser = argparse.ArgumentParser(description="Run batch inference with an existing trained model.")
    parser.add_argument("--model", default="xgboost", choices=["ridge", "random_forest", "xgboost", "neural_network"])
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    artifact = load_model(args.model)
    if not isinstance(artifact, dict):
        raise TypeError("Invalid model artifact format.")
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    frame = pd.read_csv(args.input)
    features = artifact.get("feature_columns", [])
    matrix = frame.copy()
    for column in features:
        if column not in matrix.columns:
            matrix[column] = 0.0
    predictions = artifact["model"].predict(matrix[features])
    output = frame[["race_id", "driver_id", "lap"]].copy()
    output["predicted_degradation"] = predictions
    output.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
