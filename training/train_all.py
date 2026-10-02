from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from models.compare_models import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate all supported models.")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
