import numpy as np
import pandas as pd
import pytest

from f1_tyre.evaluation.explainability import grouped_permutation_importance


class SignalModel:
    def predict(self, frame):
        return frame["signal"].to_numpy(dtype=float)


def test_grouped_permutation_reports_validation_mae_increase_not_local_shap():
    validation = pd.DataFrame(
        {
            "signal": np.arange(30, dtype=float),
            "noise": np.tile([0.0, 1.0, 2.0], 10),
        }
    )
    target = validation["signal"].to_numpy()

    report = grouped_permutation_importance(
        SignalModel(),
        validation,
        target,
        {"pace": ["signal"], "irrelevant": ["noise"]},
        n_repeats=5,
        random_state=7,
    )

    assert report.iloc[0]["feature_group"] == "pace"
    assert report.iloc[0]["importance_mean"] > 0
    assert set(report["method"]) == {"grouped_permutation_validation"}
    assert set(report["metric"]) == {"increase_in_mae"}
    assert "shap" not in " ".join(report.columns).lower()


def test_grouped_permutation_rejects_unknown_feature_group_columns():
    with pytest.raises(ValueError, match="unknown columns"):
        grouped_permutation_importance(
            SignalModel(),
            pd.DataFrame({"signal": [1.0, 2.0]}),
            [1.0, 2.0],
            {"future": ["future_lap_time"]},
        )