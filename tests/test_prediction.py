import pandas as pd

from f1_tyre.inference import predict as predict_module


def test_predict_frame_returns_shape():
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

    y_pred = predict_module.predict_with_model(X, model_name="demo")
    assert len(y_pred) == len(X)
    assert y_pred.shape[1] == 1
