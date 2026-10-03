from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1_tyre.model_loader import list_available_models, load_model, load_model_metrics
from f1_tyre.inference.predict import predict_with_model
from models.common import clean_data, create_current_features

FRONTEND_DIR = ROOT / 'frontend'
ARTIFACTS_DIR = ROOT / 'artifacts'
FEATURE_IMPORTANCE_FILE = ROOT / 'feature_importance.csv'
MODEL_METRICS_FILE = ARTIFACTS_DIR / 'model_metrics.json'


app = FastAPI(title='Synthetic Tyre-Wear Benchmark', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


class PredictionRequest(BaseModel):
    model: str = Field(default='xgboost')
    inputs: dict[str, Any] = Field(default_factory=dict)


SYNONYM_MAP = {
    'tyre_compound': 'tire_compound',
    'tire_compound': 'tire_compound',
    'compound': 'tire_compound',
    'tyre_age': 'tire_age_laps',
    'tyre_age_laps': 'tire_age_laps',
    'tire_age_laps': 'tire_age_laps',
    'lap_time': 'lap_time_sec',
    'lap_time_sec': 'lap_time_sec',
    'sector_1_time_sec': 's1_time_sec',
    'sector_2_time_sec': 's2_time_sec',
    'sector_3_time_sec': 's3_time_sec',
    'air_temperature': 'race_air_temp_c',
    'track_temperature': 'race_track_temp_c',
    'current_weather': 'weather_current',
    'current_track_status': 'track_status',
    'fuel_load': 'fuel_load_kg',
    'gap_to_leader': 'gap_to_leader_sec',
    'gap_ahead': 'gap_ahead_sec',
    'gap_behind': 'gap_behind_sec',
    'current_weather_condition': 'weather_current',
    'drivers_aggression_rating': 'driver_aggression_rating',
    'driver_aggression': 'driver_aggression_rating',
    'car_speed_rating': 'team_car_speed_rating',
    'car_downforce_rating': 'team_car_downforce_rating',
    'car_reliability_rating': 'team_car_reliability_rating',
    'pit_crew_rating': 'team_pit_crew_rating',
    'total_race_laps': 'race_total_laps',
    'current_race_lap': 'lap',
    'track_position': 'position',
    'temperature_air_c': 'race_air_temp_c',
    'temperature_track_c': 'race_track_temp_c',
    'humidity': 'race_humidity_pct',
    'circuit_country': 'circuit_country',
    'driver_nationality': 'driver_nationality',
}


def normalize_key(name: str) -> str:
    value = str(name).strip().lower().replace('-', '_').replace(' ', '_').replace('/', '_')
    value = re.sub(r'[^a-z0-9_]', '', value)
    return value


def _coerce_float(value: Any, label: str) -> float:
    if value is None or str(value).strip() == '' or str(value).strip().lower() in {'none', 'nan', 'null'}:
        raise ValueError(f'{label} is required.')
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be numeric.') from exc
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError(f'{label} must be a finite number.')
    return numeric


def _validate_basic_inputs(raw_inputs: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_inputs, dict):
        raise ValueError('Prediction payload must include a dictionary of inputs.')

    for key, value in list(raw_inputs.items()):
        if value is None or str(value).strip() == '' or str(value).strip().lower() in {'nan', 'none', 'null'}:
            raise ValueError(f"Input '{key}' cannot be empty.")

    if 'lap' in raw_inputs:
        lap = _coerce_float(raw_inputs['lap'], 'Lap')
        race_total_laps = _coerce_float(raw_inputs.get('race_total_laps', raw_inputs.get('total_race_laps', 0)), 'Total race laps') if raw_inputs.get('race_total_laps', raw_inputs.get('total_race_laps')) not in (None, '', 'nan', 'none') else None
        if race_total_laps is not None and (lap < 1 or lap > race_total_laps):
            raise ValueError('Lap must be between 1 and total race laps.')

    if 'tire_age_laps' in raw_inputs:
        tire_age = _coerce_float(raw_inputs['tire_age_laps'], 'Tyre age')
        if tire_age < 0:
            raise ValueError('Tyre age must be zero or greater.')

    if 'race_air_temp_c' in raw_inputs:
        air_temp = _coerce_float(raw_inputs['race_air_temp_c'], 'Air temperature')
        if air_temp < -40 or air_temp > 80:
            raise ValueError('Air temperature must be between -40C and 80C.')

    if 'race_track_temp_c' in raw_inputs:
        track_temp = _coerce_float(raw_inputs['race_track_temp_c'], 'Track temperature')
        if track_temp < -20 or track_temp > 90:
            raise ValueError('Track temperature must be between -20C and 90C.')

    if 'race_humidity_pct' in raw_inputs:
        humidity = _coerce_float(raw_inputs['race_humidity_pct'], 'Humidity')
        if humidity < 0 or humidity > 100:
            raise ValueError('Humidity must be between 0% and 100%.')

    return raw_inputs


def _resolve_feature_key(raw_key: str) -> str:
    lowered = normalize_key(raw_key)
    if lowered in SYNONYM_MAP:
        return SYNONYM_MAP[lowered]
    if lowered in {'current_tyrewear_pct', 'current_tyre_wear_pct', 'tire_wear_pct'}:
        return 'tire_wear_pct'
    if lowered in {'current_ers_deploy_pct', 'ers_deploy'}:
        return 'ers_deploy_pct'
    if lowered in {'current_ers_harvest_pct', 'ers_harvest'}:
        return 'ers_harvest_pct'
    if lowered in {'current_gap_to_leader', 'gap_to_leader'}:
        return 'gap_to_leader_sec'
    return lowered


def _build_legacy_feature_row(raw_inputs: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    raw_inputs = _validate_basic_inputs(raw_inputs)
    cleaned: dict[str, Any] = {}
    for key, value in raw_inputs.items():
        resolved = _resolve_feature_key(key)
        if resolved in {'driver_nationality', 'circuit_country'}:
            cleaned[resolved] = str(value).strip()
        else:
            cleaned[resolved] = value

    row: dict[str, Any] = {}
    for key, value in cleaned.items():
        if isinstance(value, str):
            if value.strip().lower() in {'nan', 'none', 'null'}:
                continue
            row[key] = value
        else:
            row[key] = value

    for feature in feature_columns:
        if feature in row:
            continue
        if feature.startswith('circuit_country_'):
            country_name = str(row.get('circuit_country', '')).strip()
            row[feature] = 1.0 if country_name and country_name.lower() == feature.replace('circuit_country_', '').lower() else 0.0
            continue
        if feature.startswith('driver_nationality_'):
            nation = str(row.get('driver_nationality', '')).strip()
            row[feature] = 1.0 if nation and nation.lower() == feature.replace('driver_nationality_', '').lower() else 0.0
            continue

        if any(feature.endswith(suffix) for suffix in ('_lag1', '_lag2', '_lag3')):
            base_name = feature.rsplit('_lag', 1)[0]
            if base_name in row:
                row[feature] = row[base_name]
                continue
        if any(feature.endswith(suffix) for suffix in ('_roll3_mean', '_roll5_mean')):
            base_name = feature.rsplit('_roll', 1)[0]
            if base_name in row:
                row[feature] = row[base_name]
                continue
        if feature.endswith('_roll5_std') or feature.endswith('_roll3_std'):
            row[feature] = 0.0
            continue
        if feature.endswith('_std'):
            row[feature] = 0.0
            continue
        if feature.endswith('_mean'):
            base_name = feature.rsplit('_mean', 1)[0]
            row[feature] = row.get(base_name, 0.0)
            continue
        if feature.startswith('lap_time_sec_') and 'lap_time_sec' in row:
            row[feature] = row['lap_time_sec']
            continue
        if feature in {'lap_time_per_km', 'tire_age_ratio', 'fuel_ratio', 'total_nearest_gap', 'position_change', 'race_progress', 'car_performance_index'}:
            row[feature] = 0.0
            continue
        row[feature] = 0.0

    for feature in feature_columns:
        if feature in {'season', 'race_round', 'circuit_id', 'driver_id', 'team_id', 'grid_position', 'race_total_laps', 'lap', 'position'}:
            row[feature] = _coerce_float(row.get(feature, 0), feature)
        elif feature not in {'tire_compound', 'track_status', 'weather_current', 'circuit_country', 'driver_nationality'}:
            if isinstance(row.get(feature), str):
                val = row.get(feature)
                row[feature] = _coerce_float(val, feature)
    return row


def _safe_execute_prediction(model_name: str, frame: pd.DataFrame):
    try:
        artifact = load_model(model_name)
    except ValueError:
        available = list_available_models()
        resolved = available[0] if available else 'xgboost'
        artifact = load_model(resolved)
    if not isinstance(artifact, dict):
        raise TypeError('Model artifact is not in the expected dictionary format.')
    model = artifact.get('model')
    if model is None:
        raise KeyError('The selected model artifact does not include a trained estimator.')
    feature_columns = artifact.get('feature_columns', [])
    frame = create_current_features(clean_data(frame))
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input frame is missing required model features: {missing}")
    return artifact, frame[feature_columns]


@app.get('/api/health')
def health() -> dict[str, Any]:
    return {
        'status': 'ok',
        'project': 'Synthetic Tyre-Wear Benchmark',
        'available_models': list_available_models(),
    }


@app.get('/api/models')
def get_models() -> dict[str, Any]:
    models = list_available_models()
    return {'models': models, 'default_model': 'xgboost'}


@app.get('/api/model-metrics')
def get_model_metrics() -> dict[str, Any]:
    if MODEL_METRICS_FILE.exists():
        with MODEL_METRICS_FILE.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return {'models': payload.get('models', payload.get('results', []))}
        if isinstance(payload, list):
            return {'models': payload}
    return {'models': []}


@app.get('/api/feature-importance')
def get_feature_importance() -> dict[str, Any]:
    if not FEATURE_IMPORTANCE_FILE.exists():
        return {
            'features': [],
            'method': 'unavailable',
            'interpretation': 'No predictive feature-importance data is available.',
        }

    with FEATURE_IMPORTANCE_FILE.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        features = []
        methods = set()
        for row in reader:
            method = row.get('method', '').strip()
            if method:
                methods.add(method)
            record = {
                'feature': row.get('feature', '').strip(),
                'importance': float(row.get('importance', 0.0) or 0.0),
            }
            if record['feature']:
                features.append(record)
    return {
        'features': features[:30],
        'count': len(features),
        'method': next(iter(methods)) if len(methods) == 1 else 'legacy_unspecified',
        'interpretation': 'Global predictive association only; not causal importance or a local attribution.',
    }


@app.post('/api/explain')
def explain_prediction(payload: PredictionRequest) -> dict[str, Any]:
    model_name = payload.model.strip().lower()
    if model_name not in list_available_models():
        raise HTTPException(status_code=422, detail=f'Invalid model name: {payload.model}. Available models: {list_available_models()}')

    artifact = load_model(model_name)
    feature_columns = artifact.get('feature_columns', [])
    try:
        row = _build_legacy_feature_row(payload.inputs, feature_columns)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    importance_response = get_feature_importance()
    feature_associations = [
        {
            'feature': feature['feature'],
            'input_value': row.get(feature['feature']),
            'predictive_importance': feature['importance'],
        }
        for feature in importance_response['features'][:10]
    ]

    return {
        'model': model_name,
        'explanation_method': 'global_predictive_importance_association',
        'importance_method': importance_response['method'],
        'local_attribution_computed': False,
        'feature_associations': feature_associations,
        'notes': 'Global predictive importance is shown beside input values for context. It is not a local contribution or causal effect; no SHAP values are computed.',
    }


@app.post('/api/predict')
def predict(payload: PredictionRequest) -> dict[str, Any]:
    model_name = payload.model.strip().lower()
    if model_name not in list_available_models():
        raise HTTPException(status_code=422, detail=f'Invalid model name: {payload.model}. Available models: {list_available_models()}')

    try:
        artifact = load_model(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail='Prediction could not be generated because the trained model artifact is unavailable.') from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    feature_columns = artifact.get('feature_columns', [])
    if not feature_columns:
        raise HTTPException(status_code=503, detail='The selected model artifact does not expose a usable feature list.')

    try:
        row = _build_legacy_feature_row(payload.inputs, feature_columns)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    frame = pd.DataFrame([row])
    try:
        artifact, matrix = _safe_execute_prediction(model_name, frame)
        prediction = float(artifact['model'].predict(matrix)[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f'Prediction could not be generated: {exc}') from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=500, detail='Inference failed while using the saved model artifact.') from exc

    result = {
        'model': model_name,
        'prediction': prediction,
        'prediction_pct': prediction,
        'inputs': row,
        'status': 'ok',
    }
    if 'tire_wear_pct' in row and isinstance(row.get('tire_wear_pct'), (int, float)):
        result['next_lap_wear_estimate'] = float(row['tire_wear_pct']) + prediction
    return result


@app.get('/')
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / 'index.html')


app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('backend.app:app', host='0.0.0.0', port=8000, reload=False)
