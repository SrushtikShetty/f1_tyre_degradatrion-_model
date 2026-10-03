from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import pandas as pd

from f1_tyre.data import build_target_frame
from f1_tyre.evaluation.baselines import fit_baseline_models, score_baselines
from f1_tyre.evaluation.splits import chronological_race_splits


DEFAULT_DATASET = PROJECT_ROOT / "data" / "generated" / "authoritative_seed_20261004.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "evaluation" / "validation_baselines_seed_20261004.json"


def run(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_path: str | Path = DEFAULT_OUTPUT,
    seed: int = 20261004,
) -> dict[str, Any]:
    source = Path(dataset_path)
    if not source.exists():
        raise FileNotFoundError(f"Generated dataset not found: {source}. Run data_generation.generate_dataset first.")

    raw = pd.read_csv(source)
    labeled = build_target_frame(raw)
    partitions, protocol = chronological_race_splits(labeled)
    fitted = fit_baseline_models(partitions["train"])
    metrics = score_baselines(fitted, partitions["validation"])
    try:
        display_path = source.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(source.resolve())

    payload: dict[str, Any] = {
        "status": "VALIDATION_ONLY_DEVELOPMENT_BASELINES",
        "dataset": {
            "type": "synthetic",
            "path": display_path,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "seed": int(seed),
            "rows": int(len(labeled)),
        },
        "target": "next_lap_wear_increment: simulated wear realized during lap t+1 from state available at the end of lap t",
        "baseline_fit_scope": "train only",
        "evaluation_protocol": protocol,
        "final_holdout_scored": False,
        "metrics": metrics,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit simple baselines on TRAIN and score VALIDATION only.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    run(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()