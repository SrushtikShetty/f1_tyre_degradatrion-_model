from __future__ import annotations

import pandas as pd

from f1_tyre.config import FORBIDDEN_LEAKAGE_PATTERNS


def audit_feature_matrix(df: pd.DataFrame) -> bool:
    """Fail loudly if forbidden future-state columns slip into the feature matrix."""
    columns = [str(column).lower() for column in df.columns]
    matches = []
    for column in columns:
        if any(pattern in column for pattern in FORBIDDEN_LEAKAGE_PATTERNS):
            matches.append(column)
    if matches:
        raise ValueError(
            "Leakage detected in feature matrix. Forbidden columns: " + ", ".join(sorted(set(matches)))
        )
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
