from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from f1_tyre.original_derived_benchmark import build_original_derived_dataset, build_original_derived_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = PROJECT_ROOT / "train.csv"
DEFAULT_TEST = PROJECT_ROOT / "test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "derived"
PART_PREFIX = "original_derived_seed"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_parts(data_dir: str | Path, seed: int = 42) -> list[Path]:
    data_dir = Path(data_dir)
    return sorted(data_dir.glob(f"{PART_PREFIX}_{seed}_part*.csv"))


def load_split_benchmark(data_dir: str | Path = DEFAULT_OUTPUT_DIR, seed: int = 42) -> pd.DataFrame:
    data_dir = Path(data_dir)
    part_paths = _dataset_parts(data_dir, seed=seed)
    single_csv = data_dir / f"{PART_PREFIX}_{seed}.csv"
    if not part_paths and single_csv.exists():
        return pd.read_csv(single_csv)
    if not part_paths:
        raise FileNotFoundError(f"No benchmark CSV was found under {data_dir} for seed {seed}.")
    return pd.concat([pd.read_csv(path) for path in part_paths], ignore_index=True)


def _write_split_dataset(frame: pd.DataFrame, output_dir: str | Path, seed: int = 42) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_count = len(frame)
    split_count = 2 if row_count > 0 else 0
    part_paths: list[Path] = []
    start = 0
    for part_index in range(1, split_count + 1):
        if part_index == split_count:
            end = row_count
        else:
            end = start + (row_count // split_count)
        chunk = frame.iloc[start:end].copy()
        path = out_dir / f"{PART_PREFIX}_{seed}_part{part_index}.csv"
        chunk.to_csv(path, index=False)
        part_paths.append(path)
        start = end
    return part_paths


def build_benchmark(train_path: str | Path = DEFAULT_TRAIN, test_path: str | Path = DEFAULT_TEST, seed: int = 42, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict:
    train_path = Path(train_path).expanduser().resolve()
    test_path = Path(test_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    derived = build_original_derived_dataset(train, test, seed=seed)
    part_paths = _write_split_dataset(derived, out_dir, seed=seed)
    manifest_path = out_dir / f"{PART_PREFIX}_{seed}_manifest.json"

    manifest = build_original_derived_manifest(train, test, seed=seed)
    relative_paths = [path.relative_to(PROJECT_ROOT).as_posix() for path in part_paths]
    manifest["dataset_path"] = relative_paths[0]
    manifest["dataset_files"] = relative_paths
    manifest["split_row_counts"] = {path.name: int(pd.read_csv(path).shape[0]) for path in part_paths}
    manifest["source_train_sha256"] = _sha256(train_path)
    manifest["source_test_sha256"] = _sha256(test_path)
    manifest["part_sha256"] = {path.name: _sha256(path) for path in part_paths}
    merged_bytes = b"".join(path.read_bytes() for path in part_paths)
    manifest["combined_dataset_sha256"] = hashlib.sha256(merged_bytes).hexdigest()
    manifest["derived_dataset_sha256"] = manifest["combined_dataset_sha256"]
    manifest["dataset_hash"] = f"sha256:{manifest['combined_dataset_sha256']}"
    manifest["notes"] = (
        "The original benchmark was split across multiple CSV files because GitHub rejects individual files above 100 MB. "
        "Each split file contains the same columns and deterministic row ordering from the original derived dataset, and the parts can be concatenated to recover the exact benchmark."
    )
    manifest["github_file_size_note"] = "The original single CSV exceeded GitHub's 100 MB per-file limit; the benchmark is therefore stored as deterministic split files under 100 MB each."
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an original-data-derived semi-synthetic benchmark from the supplied train/test CSVs.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = build_benchmark(args.train, args.test, seed=args.seed, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
