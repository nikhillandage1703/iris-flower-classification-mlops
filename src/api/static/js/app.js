/**
 * Iris Flower Classification - Frontend Application Logic
 * Communicates with FastAPI backend REST endpoints.
 */

// Tab Navigation
const navLinks = document.querySelectorAll('.nav-link');
const tabSections = document.querySelectorAll('.tab-section');
const pageTitle = document.getElementById('page-title');
const pageSubtitle = document.getElementById('page-subtitle');

const tabMeta = {
  overview: {
    title: 'Dashboard Overview',
    subtitle: 'Real-time telemetry, model metrics, and experiment tracking',
  },
  dataset: {
    title: 'Dataset Explorer',
    subtitle: 'Descriptive statistics, distribution properties, and EDA artifacts',
  },
  comparison: {
    title: 'Model Comparison',
    subtitle: 'Empirical benchmark between Decision Tree and Random Forest',
  },
  evaluation: {
    title: 'Model Evaluation',
    subtitle: 'Confusion matrices, error analysis, and multiclass averaging',
  },
  prediction: {
    title: 'Flower Prediction Lab',
    subtitle: 'Interactive inference engine with probability confidence scores',
  },
  pipeline: {
    title: 'MLOps Pipeline Architecture',
    subtitle: 'End-to-end workflow from data ingestion to containerized serving',
  },
  about: {
    title: 'About the Project',
    subtitle: 'Academic objectives, architecture design, and future enhancements',
  },
};

function switchToTab(tabName) {
  navLinks.forEach((link) => {
    if (link.dataset.tab === tabName) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  tabSections.forEach((section) => {
    if (section.id === `tab-${tabName}`) {
      section.classList.add('active');
    } else {
      section.classList.remove('active');
    }
  });

  if (tabMeta[tabName]) {
    pageTitle.textContent = tabMeta[tabName].title;
    pageSubtitle.textContent = tabMeta[tabName].subtitle;
  }
}

navLinks.forEach((link) => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const tabName = link.dataset.tab;
    switchToTab(tabName);
  });
});

// Theme Toggle
const themeToggleBtn = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const themeLabel = document.getElementById('theme-label');

themeToggleBtn.addEventListener('click', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const targetTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', targetTheme);

  if (targetTheme === 'light') {
    themeIcon.textContent = '🌙';
    themeLabel.textContent = 'Dark Mode';
  } else {
    themeIcon.textContent = '☀️';
    themeLabel.textContent = 'Light Mode';
  }
});

// Input Synchronization (Sliders <-> Number Inputs)
function syncInputs(prefix, fromSlider = true) {
  const slider = document.getElementById(`slider-${prefix}`);
  const number = document.getElementById(`num-${prefix}`);

  if (fromSlider) {
    number.value = slider.value;
  } else {
    slider.value = number.value;
  }
}

function applyPreset(sl, sw, pl, pw) {
  document.getElementById('slider-sepal-length').value = sl;
  document.getElementById('num-sepal-length').value = sl;
  document.getElementById('slider-sepal-width').value = sw;
  document.getElementById('num-sepal-width').value = sw;
  document.getElementById('slider-petal-length').value = pl;
  document.getElementById('num-petal-length').value = pl;
  document.getElementById('slider-petal-width').value = pw;
  document.getElementById('num-petal-width').value = pw;

  // Auto trigger prediction on preset select
  document.getElementById('prediction-form').dispatchEvent(new Event('submit'));
}

function resetForm() {
  applyPreset(5.1, 3.5, 1.4, 0.2);
  document.getElementById('result-content').style.display = 'none';
  document.getElementById('result-empty').style.display = 'block';
}

// Prediction Handler
async function handlePredictionSubmit(event) {
  event.preventDefault();

  const btn = document.getElementById('btn-predict');
  const originalText = btn.innerHTML;
  btn.innerHTML = '<span>⏳ Computing...</span>';
  btn.disabled = true;

  const payload = {
    sepal_length: parseFloat(document.getElementById('num-sepal-length').value),
    sepal_width: parseFloat(document.getElementById('num-sepal-width').value),
    petal_length: parseFloat(document.getElementById('num-petal-length').value),
    petal_width: parseFloat(document.getElementById('num-petal-width').value),
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Inference error');
    }

    const data = await res.json();
    renderPredictionResult(data);
  } catch (error) {
    alert(`Prediction Failed: ${error.message}`);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

function renderPredictionResult(data) {
  document.getElementById('result-empty').style.display = 'none';
  const content = document.getElementById('result-content');
  content.style.display = 'block';

  const badge = document.getElementById('result-species-badge');
  badge.textContent = data.predicted_species;
  badge.className = 'species-badge';

  const icon = document.getElementById('result-flower-icon');
  if (data.predicted_species === 'Iris-setosa') {
    badge.classList.add('badge-setosa');
    icon.textContent = '🌺';
  } else if (data.predicted_species === 'Iris-versicolor') {
    badge.classList.add('badge-versicolor');
    icon.textContent = '🌼';
  } else {
    badge.classList.add('badge-virginica');
    icon.textContent = '🌸';
  }

  document.getElementById('result-class-idx').textContent = data.predicted_class;
  document.getElementById('result-model-name').textContent = data.model_name;

  // Update Probabilities
  const probs = data.probabilities || {};
  const pSetosa = Math.round((probs['Iris-setosa'] || 0) * 100);
  const pVersicolor = Math.round((probs['Iris-versicolor'] || 0) * 100);
  const pVirginica = Math.round((probs['Iris-virginica'] || 0) * 100);

  document.getElementById('prob-val-setosa').textContent = `${pSetosa}%`;
  document.getElementById('prob-bar-setosa').style.width = `${pSetosa}%`;

  document.getElementById('prob-val-versicolor').textContent = `${pVersicolor}%`;
  document.getElementById('prob-bar-versicolor').style.width = `${pVersicolor}%`;

  document.getElementById('prob-val-virginica').textContent = `${pVirginica}%`;
  document.getElementById('prob-bar-virginica').style.width = `${pVirginica}%`;
}

// Fetch Initial Telemetry and Dashboard Data
let overviewChartInstance = null;
let multiMetricChartInstance = null;

async function fetchTelemetry() {
  // 1. Health
  try {
    const healthRes = await fetch('/health');
    const health = await healthRes.json();
    const dot = document.getElementById('api-status-dot');
    const txt = document.getElementById('api-status-text');
    if (health.status === 'healthy') {
      dot.style.backgroundColor = 'var(--accent-emerald)';
      txt.textContent = `Online (${health.model_name})`;
    } else {
      dot.style.backgroundColor = 'var(--accent-amber)';
      txt.textContent = 'Degraded';
    }
  } catch (e) {
    document.getElementById('api-status-dot').style.backgroundColor = 'var(--accent-rose)';
    document.getElementById('api-status-text').textContent = 'API Offline';
  }

  // 2. Model Info
  try {
    const infoRes = await fetch('/model-info');
    if (infoRes.ok) {
      const info = await infoRes.json();
      document.getElementById('stat-accuracy').textContent = `${(info.best_model_metrics.accuracy * 100).toFixed(1)}%`;
      document.getElementById('meta-algo').textContent = info.algorithm;
      document.getElementById('meta-name').textContent = info.best_model_display;
      document.getElementById('meta-exp').textContent = info.mlflow_experiment;
      document.getElementById('meta-run-id').textContent = info.mlflow_run_id || 'Active Run';
      document.getElementById('meta-f1').textContent = (info.best_model_metrics.f1_score_weighted * 100).toFixed(2) + '%';
      document.getElementById('meta-time').textContent = new Date(info.training_timestamp).toLocaleString();
    }
  } catch (e) {
    console.warn('Model info load error:', e);
  }

  // 3. Metrics Comparison
  try {
    const metricsRes = await fetch('/api/metrics');
    if (metricsRes.ok) {
      const metrics = await metricsRes.json();
      renderComparisonTable(metrics);
      renderCharts(metrics);
    }
  } catch (e) {
    console.warn('Metrics load error:', e);
  }

  // 4. Dataset Preview
  try {
    const datasetRes = await fetch('/api/dataset-stats');
    if (datasetRes.ok) {
      const ds = await datasetRes.json();
      renderDatasetPreview(ds.preview);
    }
  } catch (e) {
    console.warn('Dataset stats load error:', e);
  }
}

function renderComparisonTable(metrics) {
  const dt = metrics.decision_tree.metrics;
  const rf = metrics.random_forest.metrics;
  const best = metrics.best_model;

  const dtBadge = best === 'decision_tree' 
    ? '<span style="color: var(--accent-emerald); font-weight:700;">★ Best Model</span>' 
    : '<span style="color: var(--text-muted);">Baseline</span>';
  
  const rfBadge = best === 'random_forest' 
    ? '<span style="color: var(--accent-emerald); font-weight:700;">★ Production Selected</span>' 
    : '<span style="color: var(--text-muted);">Comparison</span>';

  const html = `
    <tr>
      <td><strong>Decision Tree Classifier</strong></td>
      <td>${(dt.train_accuracy * 100).toFixed(1)}%</td>
      <td><strong>${(dt.test_accuracy * 100).toFixed(1)}%</strong></td>
      <td>${(dt.test_precision_weighted * 100).toFixed(1)}%</td>
      <td>${(dt.test_recall_weighted * 100).toFixed(1)}%</td>
      <td><strong>${(dt.test_f1_weighted * 100).toFixed(1)}%</strong></td>
      <td>${dtBadge}</td>
    </tr>
    <tr>
      <td><strong>Random Forest Classifier</strong></td>
      <td>${(rf.train_accuracy * 100).toFixed(1)}%</td>
      <td><strong>${(rf.test_accuracy * 100).toFixed(1)}%</strong></td>
      <td>${(rf.test_precision_weighted * 100).toFixed(1)}%</td>
      <td>${(rf.test_recall_weighted * 100).toFixed(1)}%</td>
      <td><strong>${(rf.test_f1_weighted * 100).toFixed(1)}%</strong></td>
      <td>${rfBadge}</td>
    </tr>
  `;

  document.getElementById('comparison-table-body').innerHTML = html;
}

function renderDatasetPreview(samples) {
  if (!samples || !samples.length) return;
  const rows = samples.map((row, idx) => `
    <tr>
      <td>${idx + 1}</td>
      <td>${row.sepal_length.toFixed(1)}</td>
      <td>${row.sepal_width.toFixed(1)}</td>
      <td>${row.petal_length.toFixed(1)}</td>
      <td>${row.petal_width.toFixed(1)}</td>
      <td><span style="font-family: monospace;">${row.target}</span></td>
      <td><strong>${row.species_name}</strong></td>
    </tr>
  `).join('');
  document.getElementById('preview-table-body').innerHTML = rows;
}

function renderCharts(metrics) {
  if (typeof Chart === 'undefined') return;

  const dt = metrics.decision_tree.metrics;
  const rf = metrics.random_forest.metrics;

  // 1. Overview Bar Chart
  const ctxOverview = document.getElementById('overview-chart').getContext('2d');
  if (overviewChartInstance) overviewChartInstance.destroy();
  overviewChartInstance = new Chart(ctxOverview, {
    type: 'bar',
    data: {
      labels: ['Decision Tree', 'Random Forest'],
      datasets: [{
        label: 'Test Accuracy (%)',
        data: [dt.test_accuracy * 100, rf.test_accuracy * 100],
        backgroundColor: ['#3b82f6', '#10b981'],
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 80,
          max: 100,
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(148, 163, 184, 0.1)' }
        },
        x: {
          ticks: { color: '#94a3b8' },
          grid: { display: false }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  // 2. Multi-Metric Comparison Chart
  const ctxMulti = document.getElementById('multi-metric-chart').getContext('2d');
  if (multiMetricChartInstance) multiMetricChartInstance.destroy();
  multiMetricChartInstance = new Chart(ctxMulti, {
    type: 'bar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
      datasets: [
        {
          label: 'Decision Tree',
          data: [
            dt.test_accuracy * 100,
            dt.test_precision_weighted * 100,
            dt.test_recall_weighted * 100,
            dt.test_f1_weighted * 100,
          ],
          backgroundColor: '#3b82f6',
          borderRadius: 4,
        },
        {
          label: 'Random Forest',
          data: [
            rf.test_accuracy * 100,
            rf.test_precision_weighted * 100,
            rf.test_recall_weighted * 100,
            rf.test_f1_weighted * 100,
          ],
          backgroundColor: '#10b981',
          borderRadius: 4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 85,
          max: 100,
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(148, 163, 184, 0.1)' }
        },
        x: {
          ticks: { color: '#94a3b8' },
          grid: { display: false }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#94a3b8' }
        }
      }
    }
  });
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  fetchTelemetry();
});
