const state = {
  models: [],
  metrics: [],
  featureImportance: [],
  prediction: null,
  explanation: [],
};

const form = document.getElementById('predictionForm');
const modelSelect = document.getElementById('modelSelect');
const submitButton = document.getElementById('submitButton');
const formStatus = document.getElementById('formStatus');

function showStatus(message, isError = false) {
  formStatus.textContent = message;
  formStatus.dataset.state = isError ? 'error' : message ? 'success' : '';
}

class UserFacingError extends Error {}

function modelLabel(modelName) {
  const labels = {
    xgboost: 'XGBoost',
    random_forest: 'Random Forest',
    neural_network: 'Neural Network',
    ridge: 'Ridge',
  };
  return labels[String(modelName).toLowerCase()] || String(modelName).replaceAll('_', ' ');
}

function requestErrorMessage(status, detail) {
  if (status === 422) {
    const message = String(detail || '');
    if (/^Lap must be between/i.test(message)) {
      return 'Current lap must be between 1 and total race laps.';
    }
    if (/^Tyre age must be zero or greater/i.test(message)) {
      return 'Tyre age must be zero or greater.';
    }
    if (/^Air temperature must be between/i.test(message)) {
      return 'Air temperature must be between -40°C and 80°C.';
    }
    if (/^Track temperature must be between/i.test(message)) {
      return 'Track temperature must be between -20°C and 90°C.';
    }
    if (/^Humidity must be between/i.test(message)) {
      return 'Humidity must be between 0% and 100%.';
    }
    return 'Some race inputs are invalid. Review the highlighted values and try again.';
  }
  if (status === 503) {
    return 'The selected saved model is unavailable right now.';
  }
  return 'The prediction service could not complete this request. Try again shortly.';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[character]);
}

function setLoadingState(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? 'Running inference...' : 'Predict Next-Lap Degradation';
  submitButton.setAttribute('aria-busy', String(isLoading));
}

function normalizePayloadValue(key, value) {
  if (value === undefined || value === null || String(value).trim() === '') {
    return null;
  }

  const cleaned = String(value).trim();
  const numericKeys = [
    'season', 'race_round', 'circuit_id', 'driver_id', 'team_id', 'grid_position', 'lap', 'position',
    'race_total_laps', 'tire_age_laps', 'tire_wear_pct', 'lap_time_sec', 's1_time_sec', 's2_time_sec',
    's3_time_sec', 'gap_to_leader_sec', 'gap_ahead_sec', 'gap_behind_sec', 'fuel_load_kg', 'ers_deploy_pct',
    'ers_harvest_pct', 'drs_activated', 'race_air_temp_c', 'race_track_temp_c', 'race_humidity_pct',
    'driver_skill_rating', 'driver_aggression_rating', 'driver_consistency_rating', 'team_car_speed_rating',
    'team_car_downforce_rating', 'team_car_reliability_rating', 'team_pit_crew_rating', 'team_budget_tier',
    'circuit_length_km', 'circuit_turns', 'circuit_drs_zones', 'circuit_overtake_difficulty', 'circuit_base_lap_time_sec',
  ];

  if (numericKeys.includes(key)) {
    const numeric = Number(cleaned);
    return Number.isFinite(numeric) ? numeric : null;
  }

  return cleaned;
}

function buildPayload() {
  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = normalizePayloadValue(key, value);
  }

  if (payload.circuit && !payload.circuit_country) {
    payload.circuit_country = payload.circuit;
  }
  if (payload.driver && !payload.driver_nationality) {
    payload.driver_nationality = payload.driver;
  }

  return {
    model: modelSelect.value,
    inputs: payload,
  };
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new UserFacingError(requestErrorMessage(response.status, payload.detail || payload.error));
  }
  return payload;
}

async function initializeDashboard() {
  try {
    const modelsResponse = await fetchJson('/api/models');
    const metricsResponse = await fetchJson('/api/model-metrics');
    const featureResponse = await fetchJson('/api/feature-importance');

    state.models = modelsResponse.models || [];
    state.metrics = metricsResponse.models || [];
    state.featureImportance = featureResponse.features || [];

    const appStatus = document.getElementById('appStatusText');
    if (appStatus) {
      appStatus.textContent = 'Saved models ready';
    }

    renderModelOptions(state.models);
    renderModelComparisonChart(state.metrics);
    renderImportanceChart(state.featureImportance);
    renderPredictionCharts();
    renderMetricsTable(state.metrics);
    renderFeatureImportanceTable(state.featureImportance);
  } catch (error) {
    const appStatus = document.getElementById('appStatusText');
    if (appStatus) {
      appStatus.textContent = 'Model service unavailable';
      appStatus.parentElement.dataset.state = 'error';
    }
    showStatus('Model information could not be loaded. Refresh the page and try again.', true);
  }
}

function renderModelOptions(models) {
  modelSelect.innerHTML = '';
  models.forEach((modelName) => {
    const option = document.createElement('option');
    option.value = modelName;
    option.textContent = modelName.replace('_', ' ');
    option.selected = modelName === 'xgboost';
    modelSelect.appendChild(option);
  });
}

function renderMetricsCards(metrics) {
  const container = document.getElementById('metricsCards');
  if (!container) {
    return;
  }
  const entries = Array.isArray(metrics) ? metrics : [];
  const cards = entries.map((entry) => {
    const modelName = entry.model || entry.Model || 'Model';
    const displayName = modelLabel(modelName);
    const r2 = Number(entry.r2 ?? entry.R2 ?? 0).toFixed(4);
    const mae = Number(entry.mae ?? entry.MAE ?? 0).toFixed(4);
    const rmse = Number(entry.rmse ?? entry.RMSE ?? 0).toFixed(4);
    const selected = modelName.toLowerCase() === modelSelect.value.toLowerCase();
    return `
      <article class="metric-card ${selected ? 'highlight' : ''}">
        <span class="label">${escapeHtml(displayName)}</span>
        <strong>R² ${r2}</strong>
        <div>MAE ${mae}</div>
        <div>RMSE ${rmse}</div>
      </article>
    `;
  }).join('');
  container.innerHTML = cards || '<div class="metric-card"><span class="label">No metrics</span><strong>Unavailable</strong></div>';
}

function renderMetricsTable(metrics) {
  const tbody = document.getElementById('metricsTableBody');
  if (!tbody) {
    return;
  }
  tbody.innerHTML = (metrics || []).map((entry) => {
    const name = entry.model || entry.Model || 'Model';
    const r2 = Number(entry.r2 ?? entry.R2 ?? 0).toFixed(4);
    const mae = Number(entry.mae ?? entry.MAE ?? 0).toFixed(4);
    const rmse = Number(entry.rmse ?? entry.RMSE ?? 0).toFixed(4);
    return `
      <tr>
        <td>${escapeHtml(modelLabel(name))}</td>
        <td>${r2}</td>
        <td>${mae}</td>
        <td>${rmse}</td>
      </tr>
    `;
  }).join('');
  renderMetricsCards(metrics);
}

function renderFeatureImportanceTable(features) {
  const wrapper = document.getElementById('localExplanation');
  if (!wrapper) {
    return;
  }
  const entries = (features || []).slice(0, 6).map((feature) => {
    const label = String(feature.feature || feature.name || 'Feature').replaceAll('_', ' ');
    const importance = Number(feature.importance || 0).toFixed(4);
    return `
      <div class="explanation-item">
        <div>
          <strong>${escapeHtml(label)}</strong>
          <span>Predictive importance</span>
        </div>
        <span>${importance}</span>
      </div>
    `;
  }).join('');
  wrapper.innerHTML = entries || '<p class="result-copy">No feature importance data is available yet.</p>';
}

function formatPrediction(value) {
  const numericValue = Number(value || 0);
  if (Number.isNaN(numericValue)) {
    return '--';
  }
  const prefix = numericValue >= 0 ? '+' : '-';
  return `${prefix}${Math.abs(numericValue).toFixed(2)} pp`;
}

function renderPrediction(result) {
  const prediction = Number(result.prediction ?? 0);
  const valueElement = document.getElementById('predictionValue');
  const interpretation = document.getElementById('predictionInterpretation');
  const tyreAge = document.getElementById('resultTyreAge');
  const compound = document.getElementById('resultCompound');
  const selectedModel = document.getElementById('resultModel');
  const nextLapWear = document.getElementById('resultNextLapWear');
  const heroModel = document.getElementById('heroModelLabel');
  const heroPrediction = document.getElementById('heroPredictionValue');
  const aggressionValue = document.getElementById('heroAggressionValue');
  const inputs = result.inputs || {};
  const aggression = Number(inputs.driver_aggression_rating ?? 0);

  valueElement.textContent = formatPrediction(prediction);
  heroPrediction.textContent = formatPrediction(prediction);
  tyreAge.textContent = `${inputs.tire_age_laps ?? '--'} laps`;
  compound.textContent = inputs.tire_compound || '--';
  selectedModel.textContent = result.model || modelSelect.value;
  nextLapWear.textContent = Number.isFinite(Number(result.next_lap_wear_estimate))
    ? `${Number(result.next_lap_wear_estimate).toFixed(2)}%`
    : 'Not available';
  heroModel.textContent = modelLabel(result.model || modelSelect.value);
  aggressionValue.textContent = `${aggression}`;

  interpretation.textContent = `The model predicts a degradation increment of ${Math.abs(prediction).toFixed(2)} percentage points for the next lap.`;

  const aggressionGauge = document.querySelector('.gauge-ring');
  const fillDegrees = Math.min(320, Math.max(30, aggression * 3.1));
  if (aggressionGauge) {
    aggressionGauge.style.background = `conic-gradient(var(--accent) 0deg ${fillDegrees}deg, rgba(255,255,255,0.06) ${fillDegrees}deg 360deg)`;
  }

  document.getElementById('aggressionGaugeValue').textContent = Math.round(aggression);
  document.getElementById('driverAggressionScore').textContent = `${aggression}`;
  document.getElementById('driverConsistencyScore').textContent = `${inputs.driver_consistency_rating ?? '--'}`;
  document.getElementById('driverSkillScore').textContent = `${inputs.driver_skill_rating ?? '--'}`;
  document.getElementById('driverTyreAgeStat').textContent = `${inputs.tire_age_laps ?? '--'} laps`;

  document.getElementById('telemetryLap').textContent = `${inputs.lap ?? '--'}`;
  document.getElementById('telemetryCompound').textContent = `${inputs.tire_compound ?? '--'}`;
  document.getElementById('telemetryAge').textContent = `${inputs.tire_age_laps ?? '--'} laps`;
  document.getElementById('telemetryLapTime').textContent = `${inputs.lap_time_sec ?? '--'} s`;
  document.getElementById('telemetryFuel').textContent = `${inputs.fuel_load_kg ?? '--'} kg`;
  document.getElementById('telemetryTrackTemp').textContent = `${inputs.race_track_temp_c ?? '--'} C`;
  document.getElementById('telemetryAirTemp').textContent = `${inputs.race_air_temp_c ?? '--'} C`;
  document.getElementById('telemetryGap').textContent = `${inputs.gap_to_leader_sec ?? '--'} s`;
  document.getElementById('telemetryPosition').textContent = `${inputs.position ?? '--'}`;
  document.getElementById('telemetryAggression').textContent = `${aggression}`;
  renderPredictionCharts(result);
}

function validateInputs(inputs) {
  const fields = Array.from(form.querySelectorAll('[name]'));
  const clearInvalid = () => fields.forEach((field) => field.removeAttribute('aria-invalid'));
  clearInvalid();

  const invalidField = (name, message) => {
    const field = fields.find((item) => item.name === name);
    if (field) {
      field.setAttribute('aria-invalid', 'true');
      const group = field.closest('details');
      if (group) {
        group.open = true;
      }
      field.focus();
    }
    showStatus(message, true);
    return false;
  };

  const emptyField = fields.find((field) => !String(field.value).trim());
  if (emptyField) {
    return invalidField(emptyField.name, 'Complete all model inputs before running a prediction.');
  }

  const bounds = [
    ['lap', 1, Number(inputs.race_total_laps), 'Current lap must be between 1 and total race laps.'],
    ['tire_age_laps', 0, Number.POSITIVE_INFINITY, 'Tyre age must be zero or greater.'],
    ['race_air_temp_c', -40, 80, 'Air temperature must be between -40°C and 80°C.'],
    ['race_track_temp_c', -20, 90, 'Track temperature must be between -20°C and 90°C.'],
    ['race_humidity_pct', 0, 100, 'Humidity must be between 0% and 100%.'],
  ];
  for (const [name, minimum, maximum, message] of bounds) {
    const value = Number(inputs[name]);
    if (!Number.isFinite(value) || value < minimum || value > maximum) {
      return invalidField(name, message);
    }
  }
  return true;
}

async function fetchExplain(payload) {
  const wrapper = document.getElementById('localExplanation');
  if (!wrapper) {
    return true;
  }
  try {
    const explanation = await fetchJson('/api/explain', { method: 'POST', body: JSON.stringify(payload) });
    const items = explanation.feature_associations || [];
    wrapper.innerHTML = items.slice(0, 5).map((item) => `
      <div class="explanation-item">
        <div>
          <strong>${escapeHtml(String(item.feature || '').replaceAll('_', ' '))}</strong>
          <span>Input: ${escapeHtml(item.input_value ?? '--')}</span>
        </div>
        <span>${Number(item.predictive_importance ?? 0).toFixed(4)}</span>
      </div>
    `).join('') || '<p class="result-copy">No explanation data is available for this prediction.</p>';
    return true;
  } catch (error) {
    wrapper.textContent = 'Input associations are temporarily unavailable. The prediction itself is unchanged.';
    return false;
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  const payload = buildPayload();
  if (!validateInputs(payload.inputs)) {
    return;
  }
  try {
    showStatus('');
    setLoadingState(true);
    const result = await fetchJson('/api/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    state.prediction = result;
    renderPrediction(result);
    const explanationLoaded = await fetchExplain(payload);
    showStatus(explanationLoaded
      ? 'Prediction generated successfully.'
      : 'Prediction generated. Feature associations are unavailable.');
  } catch (error) {
    showStatus(error instanceof UserFacingError
      ? error.message
      : 'Prediction could not be generated. Check the inputs and try again.', true);
  } finally {
    setLoadingState(false);
  }
}

modelSelect.addEventListener('change', () => {
  document.getElementById('heroModelLabel').textContent = modelLabel(modelSelect.value);
  renderMetricsCards(state.metrics);
});

form.addEventListener('submit', handleSubmit);
form.addEventListener('input', (event) => {
  event.target.removeAttribute('aria-invalid');
  showStatus('');
});

const scrollButtons = document.querySelectorAll('[data-scroll]');
scrollButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const target = document.querySelector(button.dataset.scroll);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

initializeDashboard();
