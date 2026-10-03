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
  formStatus.style.color = isError ? '#f7b5a5' : '#f1d184';
}

function setLoadingState(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? 'Running inference...' : 'Predict Next-Lap Degradation';
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
    throw new Error(payload.detail || payload.error || 'Request failed.');
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

    renderModelOptions(state.models);
    renderModelComparisonChart(state.metrics);
    renderImportanceChart(state.featureImportance);
    renderPredictionCharts();
    renderMetricsTable(state.metrics);
    renderFeatureImportanceTable(state.featureImportance);
  } catch (error) {
    showStatus(error.message || 'The model metadata could not be loaded.', true);
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
    const r2 = Number(entry.r2 ?? entry.R2 ?? 0).toFixed(4);
    const mae = Number(entry.mae ?? entry.MAE ?? 0).toFixed(4);
    const rmse = Number(entry.rmse ?? entry.RMSE ?? 0).toFixed(4);
    const selected = modelName.toLowerCase() === modelSelect.value.toLowerCase();
    return `
      <article class="metric-card ${selected ? 'highlight' : ''}">
        <span class="label">${modelName}</span>
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
        <td>${name}</td>
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
    const label = feature.feature || feature.name || 'Feature';
    const importance = Number(feature.importance || 0).toFixed(4);
    return `
      <div class="explanation-item">
        <div>
          <strong>${label}</strong>
          <span>Global contribution</span>
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
    return '+0.00 %';
  }
  const prefix = numericValue >= 0 ? '+' : '-';
  return `${prefix}${Math.abs(numericValue).toFixed(2)} %`;
}

function renderPrediction(result) {
  const prediction = Number(result.prediction ?? 0);
  const valueElement = document.getElementById('predictionValue');
  const interpretation = document.getElementById('predictionInterpretation');
  const tyreAge = document.getElementById('resultTyreAge');
  const compound = document.getElementById('resultCompound');
  const selectedModel = document.getElementById('resultModel');
  const modelLabel = document.getElementById('heroModelLabel');
  const heroPrediction = document.getElementById('heroPredictionValue');
  const aggressionValue = document.getElementById('heroAggressionValue');
  const inputs = result.inputs || {};
  const aggression = Number(inputs.driver_aggression_rating ?? 0);

  valueElement.textContent = formatPrediction(prediction);
  heroPrediction.textContent = formatPrediction(prediction);
  tyreAge.textContent = `${inputs.tire_age_laps ?? '--'} laps`;
  compound.textContent = inputs.tire_compound || '--';
  selectedModel.textContent = result.model || modelSelect.value;
  modelLabel.textContent = (result.model || modelSelect.value).replace('_', ' ');
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
}

async function fetchExplain(payload) {
  try {
    const explanation = await fetchJson('/api/explain', { method: 'POST', body: JSON.stringify(payload) });
    const items = explanation.feature_contributions || [];
    const wrapper = document.getElementById('localExplanation');
    if (!wrapper) {
      return;
    }
    wrapper.innerHTML = items.slice(0, 5).map((item) => `
      <div class="explanation-item">
        <div>
          <strong>${item.feature}</strong>
          <span>${item.direction}</span>
        </div>
        <span>${Number(item.value ?? 0).toFixed(2)}</span>
      </div>
    `).join('') || '<p class="result-copy">No explanation data is available for this prediction.</p>';
  } catch (error) {
    showStatus(error.message || 'The local explanation endpoint could not be reached.', true);
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  const payload = buildPayload();
  try {
    showStatus('');
    setLoadingState(true);
    const result = await fetchJson('/api/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    state.prediction = result;
    renderPrediction(result);
    await fetchExplain(payload);
    showStatus('Prediction generated successfully.');
  } catch (error) {
    showStatus(error.message || 'Prediction could not be generated.', true);
  } finally {
    setLoadingState(false);
  }
}

modelSelect.addEventListener('change', () => {
  document.getElementById('heroModelLabel').textContent = modelSelect.options[modelSelect.selectedIndex].textContent;
  renderMetricsCards(state.metrics);
});

form.addEventListener('submit', handleSubmit);

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
