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
    if isinstance(payload, dict) and payload.get("metric_scope") == "supplementary_grouped_cv":
        print("SUPPLEMENTARY GROUPED CV ONLY; not a future-race performance claim.")
        print(f"Protocol: {payload.get('metric_protocol', 'unspecified')}")
        print(f"Primary chronological metrics: {payload.get('primary_metric_source', 'not recorded')}")
        print(json.dumps(payload.get("models", []), indent=2))
    else:
        print("LEGACY/UNVERIFIED METRICS: split and protocol metadata are unavailable.")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
