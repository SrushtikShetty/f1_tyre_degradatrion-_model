from __future__ import annotations

import argparse
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
from f1_tyre.evaluation.splits import chronological_race_splits
from f1_tyre.evaluation.target_diagnostics import diagnose_target_determinism


DEFAULT_DATASET = PROJECT_ROOT / "data" / "generated" / "authoritative_seed_20261004.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "evaluation" / "target_determinism_seed_20261004.json"


def run(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_path: str | Path = DEFAULT_OUTPUT,
    deterministic_demo: bool = False,
) -> dict[str, Any]:
    source = Path(dataset_path)
    labeled = build_target_frame(pd.read_csv(source))
    partitions, protocol = chronological_race_splits(labeled)
    report = diagnose_target_determinism(
        partitions["train"],
        partitions["validation"],
        evaluation_split="validation",
        deterministic_demo=deterministic_demo,
    )
    try:
        display_path = source.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(source.resolve())
    report["dataset"] = {
        "type": "synthetic",
        "path": display_path,
        "race_counts": protocol["race_counts"],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit compound-age target determinism on VALIDATION only.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deterministic-demo", action="store_true")
    args = parser.parse_args()
    run(args.input, args.output, args.deterministic_demo)


if __name__ == "__main__":
    main()