from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def main() -> None:
    path = Path("artifacts/model_metrics.json")
    if not path.exists():
        raise FileNotFoundError("Saved model metrics not found. Run explicit training first.")
    payload = json.loads(path.read_text())
    print(payload)


if __name__ == "__main__":
    main()
