import json

from fastapi.testclient import TestClient

from backend import app as backend_app
from backend.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/api/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'


def test_model_listing_endpoint():
    response = client.get('/api/models')
    assert response.status_code == 200
    payload = response.json()
    assert 'models' in payload
    assert 'xgboost' in payload['models']


def test_model_metrics_endpoint():
    response = client.get('/api/model-metrics')
    assert response.status_code == 200
    payload = response.json()
    assert 'models' in payload
    assert len(payload['models']) >= 1


def test_model_metrics_endpoint_reads_authoritative_final_holdout(monkeypatch, tmp_path):
    metrics_path = tmp_path / 'final_metrics.json'
    metrics_path.write_text(json.dumps({
        'status': 'FINAL_EVALUATION_COMPLETE',
        'dataset': {'type': 'synthetic', 'sha256': 'test-hash'},
        'target': {'definition': 'simulated next-lap increment'},
        'metrics': {'models': [{
            'model': 'xgboost',
            'validation': {'r2': 0.8, 'mae': 0.2, 'rmse': 0.3},
            'final_holdout': {'r2': 0.7, 'mae': 0.25, 'rmse': 0.35, 'number_of_rows': 12},
        }]},
    }))
    monkeypatch.setattr(backend_app, 'FINAL_METRICS_FILE', metrics_path)

    response = client.get('/api/model-metrics')

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'FINAL_EVALUATION_COMPLETE'
    assert payload['metric_scope'] == 'final_holdout'
    assert payload['dataset_type'] == 'synthetic'
    assert payload['models'] == [{
        'model': 'xgboost',
        'r2': 0.7,
        'mae': 0.25,
        'rmse': 0.35,
        'number_of_rows': 12,
        'split': 'final_holdout',
        'metric_scope': 'synthetic_final_holdout',
    }]


def test_feature_importance_endpoint():
    response = client.get('/api/feature-importance')
    assert response.status_code == 200
    payload = response.json()
    assert 'features' in payload
    assert len(payload['features']) >= 1
    assert payload['method'] == 'grouped_permutation_validation'
    assert payload['metric'] == 'increase_in_mae'
    assert payload['count'] == 8
    assert 'not a causal effect' in payload['interpretation']
    assert 'SHAP values are not computed' in payload['interpretation']


def test_explain_endpoint_does_not_claim_local_contributions_or_shap():
    response = client.post('/api/explain', json={
        'model': 'xgboost',
        'inputs': {'tire_compound': 'HARD', 'tire_age_laps': 3},
    })

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['explanation_method'] == 'global_predictive_importance_association'
    assert payload['importance_method'] == 'grouped_permutation_validation'
    assert payload['local_attribution_computed'] is False
    assert 'feature_associations' in payload
    assert 'feature_contributions' not in payload
    assert 'causal effect' in payload['notes']


def test_prediction_endpoint_uses_real_model_artifact():
    payload = {
        'model': 'xgboost',
        'inputs': {
            'season': 2024,
            'race_round': 12,
            'circuit_id': 5,
            'circuit_length_km': 5.8,
            'circuit_turns': 16,
            'circuit_drs_zones': 2,
            'circuit_overtake_difficulty': 7,
            'circuit_base_lap_time_sec': 89.5,
            'driver_id': 101,
            'driver_skill_rating': 92,
            'driver_aggression_rating': 78,
            'driver_consistency_rating': 84,
            'team_id': 9,
            'team_car_speed_rating': 88,
            'team_car_downforce_rating': 86,
            'team_car_reliability_rating': 90,
            'team_pit_crew_rating': 82,
            'race_total_laps': 58,
            'grid_position': 3,
            'q1_time_sec': 82.5,
            'q2_time_sec': 81.9,
            'q3_time_sec': 81.4,
            'lap_time_sec': 88.7,
            's1_time_sec': 28.2,
            's2_time_sec': 29.6,
            's3_time_sec': 31.0,
            'lap': 24,
            'position': 2,
            'tire_compound': 'MEDIUM',
            'tire_age_laps': 14,
            'fuel_load_kg': 76.0,
            'ers_deploy_pct': 48,
            'ers_harvest_pct': 24,
            'gap_to_leader_sec': 1.2,
            'gap_ahead_sec': 0.5,
            'gap_behind_sec': 0.8,
            'track_status': 'Green',
            'weather_current': 'Clear',
            'race_air_temp_c': 29.1,
            'race_track_temp_c': 38.2,
            'race_humidity_pct': 52,
        },
    }

    response = client.post('/api/predict', json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 'prediction' in body
    assert 'model' in body
    assert body['model'] == 'xgboost'
    assert isinstance(body['prediction'], (int, float))


def test_invalid_prediction_returns_feature_error():
    response = client.post('/api/predict', json={
        'model': 'xgboost',
        'inputs': {'lap': 0, 'tire_age_laps': -1}
    })
    assert response.status_code == 422
