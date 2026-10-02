from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    print(f"Loaded model: {args.model}")
    print("Model metadata:", artifact.get("metadata", {}))
    print("Metrics:", artifact.get("metrics", {}))


if __name__ == "__main__":
    main()
