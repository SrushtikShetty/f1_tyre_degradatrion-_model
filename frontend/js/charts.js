const chartRegistry = {};

function getChartContext(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) {
    return null;
  }
  return canvas.getContext('2d');
}

function destroyChart(key) {
  if (chartRegistry[key]) {
    chartRegistry[key].destroy();
    delete chartRegistry[key];
  }
}

function renderModelComparisonChart(metrics) {
  const ctx = getChartContext('modelComparisonChart');
  if (!ctx || typeof Chart === 'undefined' || !metrics || !metrics.length) {
    return;
  }

  const labels = metrics.map((entry) => entry.model || entry.Model || 'Model');
  const r2Values = metrics.map((entry) => Number(entry.r2 ?? entry.R2 ?? 0));

  destroyChart('modelComparison');
  chartRegistry.modelComparison = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'R² score',
        data: r2Values,
        backgroundColor: ['#ed443b', '#d08d4c', '#86cbb2', '#9baeb5'],
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (context) => `R² ${context.parsed.y.toFixed(4)}` } },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          title: { display: true, text: 'R² score', color: '#a1afb5' },
          ticks: { color: '#dfeaf1' },
          grid: { color: 'rgba(255,255,255,0.06)' },
        },
        x: {
          ticks: { color: '#dfeaf1' },
          grid: { display: false },
        },
      },
    },
  });
}

function renderImportanceChart(features) {
  const ctx = getChartContext('importanceChart');
  if (!ctx || typeof Chart === 'undefined' || !features || !features.length) {
    return;
  }

  const labels = features.slice(0, 10).map((entry) => entry.feature);
  const importances = features.slice(0, 10).map((entry) => Number(entry.importance || 0));

  destroyChart('importance');
  chartRegistry.importance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Predictive importance',
        data: importances,
        backgroundColor: '#ed443b',
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (context) => `Importance ${context.parsed.x.toFixed(4)}` } },
      },
      scales: {
        x: {
          title: { display: true, text: 'Predictive importance', color: '#a1afb5' },
          ticks: { color: '#dfeaf1' },
          grid: { color: 'rgba(255,255,255,0.06)' },
        },
        y: {
          ticks: { color: '#dfeaf1' },
          grid: { display: false },
        },
      },
    },
  });
}

function renderPredictionCharts(result) {
  const chartCanvas = document.getElementById('wearProjectionChart');
  const emptyState = document.getElementById('wearProjectionEmpty');
  const compoundTag = document.getElementById('projectionCompound');
  if (!chartCanvas || !emptyState) {
    return;
  }

  destroyChart('wearProjection');
  const inputs = result?.inputs || {};
  const currentWear = Number(inputs.tire_wear_pct);
  const nextWear = Number(result?.next_lap_wear_estimate);
  const tyreAge = Number(inputs.tire_age_laps);
  if (!Number.isFinite(currentWear) || !Number.isFinite(nextWear) || !Number.isFinite(tyreAge)) {
    chartCanvas.hidden = true;
    emptyState.hidden = false;
    emptyState.textContent = result
      ? 'A wear projection is unavailable because the response did not include both wear values.'
      : 'Run a prediction to plot the current wear and one-lap estimate.';
    if (compoundTag) {
      compoundTag.textContent = result?.inputs?.tire_compound || 'Awaiting prediction';
    }
    return;
  }

  if (typeof Chart === 'undefined') {
    chartCanvas.hidden = true;
    emptyState.hidden = false;
    emptyState.textContent = 'The chart library could not be loaded. The prediction result is still available above.';
    return;
  }

  const ctx = chartCanvas.getContext('2d');
  if (!ctx) {
    return;
  }

  chartCanvas.hidden = false;
  emptyState.hidden = true;
  if (compoundTag) {
    compoundTag.textContent = inputs.tire_compound || 'Compound not supplied';
  }

  chartRegistry.wearProjection = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [`Current · age ${tyreAge}`, `Next lap · age ${tyreAge + 1}`],
      datasets: [{
        label: 'Tyre wear',
        data: [currentWear, nextWear],
        borderColor: '#ed443b',
        backgroundColor: 'rgba(237, 68, 59, 0.12)',
        pointBackgroundColor: ['#a1afb5', '#ed443b'],
        pointBorderColor: '#11181d',
        pointBorderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 7,
        borderWidth: 2,
        tension: 0.15,
        fill: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => items[0]?.dataIndex === 1 ? 'Model estimate · next lap' : 'Submitted tyre state',
            label: (context) => `${context.parsed.y.toFixed(2)}% tyre wear`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Tyre age', color: '#a1afb5' },
          ticks: { color: '#dfeaf1' },
          grid: { display: false },
        },
        y: {
          title: { display: true, text: 'Tyre wear (%)', color: '#a1afb5' },
          min: Math.max(0, Math.floor(Math.min(currentWear, nextWear) - 3)),
          max: Math.ceil(Math.max(currentWear, nextWear) + 3),
          ticks: { color: '#dfeaf1' },
          grid: { color: 'rgba(255,255,255,0.07)' },
        },
      },
    },
  });
}
