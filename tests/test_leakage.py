import pandas as pd
import pytest

from f1_tyre.evaluation import leakage_audit


def test_leakage_audit_rejects_future_columns():
    df = pd.DataFrame({"race_id": [1], "future_tire_wear_pct": [99.0]})
    with pytest.raises(ValueError):
        leakage_audit.audit_feature_matrix(df)


def test_leakage_audit_accepts_clean_matrix():
    df = pd.DataFrame({"lap": [1], "position": [2], "fuel_load_kg": [70.0], "tire_age_laps": [3]})
    result = leakage_audit.audit_feature_matrix(df)
    assert result is True
