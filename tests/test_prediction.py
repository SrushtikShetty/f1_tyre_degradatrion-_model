import pandas as pd
import pytest

from f1_tyre.inference import predict as predict_module


class ConstantModel:
    def predict(self, frame):
        return [0.1] * len(frame)


def test_predict_frame_returns_shape(monkeypatch):
    X = pd.DataFrame(
        {
            "race_id": [1, 1, 2, 2],
            "driver_id": [1, 1, 2, 2],
            "lap": [1, 2, 1, 2],
            "tire_compound": ["C1", "C1", "C2", "C2"],
            "tire_age_laps": [0, 1, 0, 1],
            "lap_time_sec": [90.0, 92.0, 88.0, 90.0],
            "fuel_load_kg": [70.0, 68.0, 73.0, 71.0],
            "position": [2, 2, 3, 3],
        }
    )

    monkeypatch.setattr(
        predict_module,
        "load_model",
        lambda _: {"model": ConstantModel(), "feature_columns": ["lap_time_sec", "tire_age_laps"]},
    )

    y_pred = predict_module.predict_with_model(X, model_name="demo")
    assert len(y_pred) == len(X)
    assert y_pred.shape[1] == 1


def test_predict_frame_rejects_missing_model_features(monkeypatch):
    frame = pd.DataFrame({"lap_time_sec": [90.0]})
    monkeypatch.setattr(
        predict_module,
        "load_model",
        lambda _: {"model": ConstantModel(), "feature_columns": ["required_feature"]},
    )

    with pytest.raises(ValueError, match="missing required model features"):
        predict_module.predict_with_model(frame, model_name="xgboost")
