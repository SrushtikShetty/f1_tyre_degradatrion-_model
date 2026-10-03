from __future__ import annotations

import pandas as pd
from typing import Iterable, Mapping

from f1_tyre.config import FORBIDDEN_LEAKAGE_PATTERNS
from f1_tyre.strict_feature_policy import FEATURE_METADATA, FORBIDDEN_FEATURES, validate_feature_names


KNOWN_LEAKAGE_COLUMNS = {
    "tire_wear_pct",
    "future_tire_wear_pct",
    "next_lap_degradation_pct",
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


def _is_forbidden_source(name: str) -> bool:
    normalized = str(name).lower()
    return name in FORBIDDEN_FEATURES or any(pattern in normalized for pattern in FORBIDDEN_LEAKAGE_PATTERNS)


def audit_feature_matrix(
    df: pd.DataFrame,
    lineage: Mapping[str, Iterable[str]] | None = None,
) -> bool:
    """Audit feature names and registered computation dependencies before fit."""
    columns = [str(column) for column in df.columns]
    partition_columns = [
        column
        for column in columns
        if FEATURE_METADATA.get(column, {}).get("availability") == "partition_only"
    ]
    if partition_columns:
        raise ValueError("Partition identifiers cannot be model features: " + ", ".join(sorted(partition_columns)))

    matches = []
    for column in columns:
        metadata = FEATURE_METADATA.get(column)
        if (
            column in KNOWN_LEAKAGE_COLUMNS
            or _is_forbidden_source(column)
            or (metadata is not None and not metadata["allowed"])
        ):
            matches.append(column)

    if matches:
        raise ValueError(
            "Leakage detected in feature matrix. Forbidden columns: " + ", ".join(sorted(set(matches)))
        )

    validate_feature_names(columns)

    registered_lineage = lineage or {
        column: FEATURE_METADATA[column]["lineage"]
        for column in columns
    }
    lineage_errors = []
    for feature, sources in registered_lineage.items():
        if feature not in columns:
            continue
        for source in sources:
            source_name = str(source)
            source_metadata = FEATURE_METADATA.get(source_name)
            if _is_forbidden_source(source_name) or source_metadata is None:
                lineage_errors.append(f"{feature} <- {source_name}")
    if lineage_errors:
        raise ValueError("Forbidden or unregistered feature lineage: " + ", ".join(sorted(lineage_errors)))
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
