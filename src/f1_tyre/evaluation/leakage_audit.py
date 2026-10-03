from __future__ import annotations

import pandas as pd

from f1_tyre.config import FORBIDDEN_LEAKAGE_PATTERNS
from f1_tyre.strict_feature_policy import validate_feature_names


KNOWN_LEAKAGE_COLUMNS = {
    "tire_wear_pct",
    "future_tire_wear_pct",
    "next_lap_wear_increment",
    "next_lap",
    "next_tire_age",
    "next_compound",
    "pit_stop_this_lap",
    "pit_stop_duration_sec",
    "race_weather_end",
    "finish_position",
    "fastest_lap",
    "race_result",
    "status",
}


def audit_feature_matrix(df: pd.DataFrame) -> bool:
    """Fail loudly if forbidden future-state or target-proxy columns slip into the feature matrix."""
    columns = [str(column) for column in df.columns]
    matches = []
    for column in columns:
        normalized = column.lower()
        if column in KNOWN_LEAKAGE_COLUMNS or any(pattern in normalized for pattern in FORBIDDEN_LEAKAGE_PATTERNS):
            matches.append(column)

    if matches:
        raise ValueError(
            "Leakage detected in feature matrix. Forbidden columns: " + ", ".join(sorted(set(matches)))
        )

    # Enforce a strict explicit allowlist so unknown variables cannot silently leak into training.
    validate_feature_names(columns)
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit a feature matrix for leakage.")
    parser.add_argument("--input", default="data/processed/feature_matrix.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    audit_feature_matrix(df)
    print(f"Leakage audit passed for {args.input}")


if __name__ == "__main__":
    main()
