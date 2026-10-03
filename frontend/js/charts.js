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
  if (!ctx || !metrics || !metrics.length) {
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
        backgroundColor: ['#e23a3a', '#ff7b54', '#9fdbd8', '#7a93ff'],
        borderRadius: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          beginAtZero: false,
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
  if (!ctx || !features || !features.length) {
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
        label: 'Feature importance',
        data: importances,
        backgroundColor: '#e23a3a',
        borderRadius: 8,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
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

function renderEmptyChart(canvasId, title) {
  const ctx = getChartContext(canvasId);
  if (!ctx) {
    return;
  }
  destroyChart(canvasId);
  const gradient = ctx.createLinearGradient(0, 0, 0, 200);
  gradient.addColorStop(0, 'rgba(226, 58, 58, 0.18)');
  gradient.addColorStop(1, 'rgba(226, 58, 58, 0.02)');
  chartRegistry[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['No data'],
      datasets: [{
        label: title,
        data: [0],
        borderColor: 'rgba(226,58,58,0.45)',
        backgroundColor: gradient,
        borderWidth: 1,
        pointRadius: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { display: false },
      },
    },
  });
}

function renderPredictionCharts() {
  renderEmptyChart('degradationVsAgeChart', 'Predicted degradation vs tyre age');
  renderEmptyChart('lapTimeChart', 'Lap time trend');
  renderEmptyChart('degradationTrendChart', 'Tyre degradation trend');
  renderEmptyChart('aggressionVsDegradationChart', 'Driver aggression vs degradation');
}
