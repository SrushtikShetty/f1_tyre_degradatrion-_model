from f1_tyre.model_loader import list_available_models


def test_model_registry_lists_supported_models():
    models = list_available_models()
    assert "ridge" in models
    assert "random_forest" in models
    assert "xgboost" in models
    assert "neural_network" in models
