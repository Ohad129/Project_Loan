async function fetchJSON(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function showError(message) {
  const slot = document.getElementById('error-slot');
  slot.innerHTML = `<div class="error-banner">${message}</div>`;
}

function renderMetrics(metrics) {
  const cards = [
    { label: 'Accuracy', value: metrics.accuracy },
    { label: 'Precision', value: metrics.precision },
    { label: 'Recall', value: metrics.recall },
    { label: 'F1 score', value: metrics.f1_score },
  ];
  const container = document.getElementById('metric-cards');
  container.innerHTML = cards
    .map(
      (c) => `
      <div class="metric">
        <div class="label">${c.label}</div>
        <div class="value">${(c.value * 100).toFixed(1)}%</div>
      </div>`
    )
    .join('');
}

function renderConfusionMatrix(cmData) {
  const [labelNo, labelYes] = cmData.labels;
  const [[tn, fp], [fn, tp]] = cmData.matrix;
  const slot = document.getElementById('cm-slot');
  slot.innerHTML = `
    <table class="cm-table">
      <thead>
        <tr><th></th><th>Predicted ${labelNo}</th><th>Predicted ${labelYes}</th></tr>
      </thead>
      <tbody>
        <tr><th>Actual ${labelNo}</th><td class="diag">${tn}</td><td class="off">${fp}</td></tr>
        <tr><th>Actual ${labelYes}</th><td class="off">${fn}</td><td class="diag">${tp}</td></tr>
      </tbody>
    </table>
  `;
}

function renderInfo(info) {
  const rows = [
    ['Algorithm', info.algorithm],
    ['Kernel', info.kernel],
    ['Class weight', info.class_weight],
    ['Trained on', `${info.trained_rows.toLocaleString()} rows`],
    ['Tested on', `${info.test_rows.toLocaleString()} rows`],
    ['Support vectors', info.support_vectors_count.toLocaleString()],
    ['Model file', info.model_file],
    ['Target', info.target],
  ];
  const slot = document.getElementById('info-slot');
  slot.innerHTML = rows
    .map(([k, v]) => `<div class="kv-row"><span class="k">${k}</span><span class="v">${v}</span></div>`)
    .join('');
}

function renderFeatures(features) {
  document.getElementById('feature-sub').textContent =
    `${features.original_columns_used.length} original columns, expanded to ${features.encoded_features.length} model inputs`;
  const slot = document.getElementById('feature-slot');
  slot.innerHTML = features.original_columns_used
    .map((f) => `<span class="feature-tag">${f}</span>`)
    .join('');
}

function renderSamples(samples) {
  if (!samples.length) return;
  const cols = Object.keys(samples[0]);
  const table = document.getElementById('samples-table');
  const thead = `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join('')}</tr></thead>`;
  const tbody = `<tbody>${samples
    .map(
      (row) =>
        `<tr>${cols
          .map((c) => {
            if (c === 'Loan Status') {
              const approved = Number(row[c]) === 1;
              return `<td><span class="pill ${approved ? 'yes' : 'no'}">${approved ? 'Approved' : 'Denied'}</span></td>`;
            }
            return `<td>${row[c]}</td>`;
          })
          .join('')}</tr>`
    )
    .join('')}</tbody>`;
  table.innerHTML = thead + tbody;
}

function renderBoundaryChart(points) {
  const svg = document.getElementById('boundary-chart');
  const width = 480;
  const height = 320;
  const pad = 30;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);

  const scaleX = (x) => pad + ((x - xMin) / (xMax - xMin)) * (width - 2 * pad);
  const scaleY = (y) => height - pad - ((y - yMin) / (yMax - yMin)) * (height - 2 * pad);

  const circles = points
    .map((p) => {
      const color = p.label === 1 ? '#276b5c' : '#a8462f';
      return `<circle cx="${scaleX(p.x).toFixed(1)}" cy="${scaleY(p.y).toFixed(1)}" r="2.6" fill="${color}" fill-opacity="0.65" />`;
    })
    .join('');

  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="none" />
    ${circles}
    <text x="${width - 90}" y="18" font-size="11" fill="#276b5c" font-family="Inter, sans-serif">● Approved</text>
    <text x="${width - 90}" y="34" font-size="11" fill="#a8462f" font-family="Inter, sans-serif">● Denied</text>
  `;
}

async function init() {
  try {
    const [info, features, metricsData, samples, boundary] = await Promise.all([
      fetchJSON('/api/model/info'),
      fetchJSON('/api/model/features'),
      fetchJSON('/api/model/metrics'),
      fetchJSON('/api/model/samples'),
      fetchJSON('/api/model/decision_boundary'),
    ]);

    renderMetrics(metricsData.metrics);
    renderConfusionMatrix(metricsData.confusion_matrix);
    renderInfo(info);
    renderFeatures(features);
    renderSamples(samples);
    renderBoundaryChart(boundary);
  } catch (err) {
    showError(err.message || 'Could not load model data. Have you run train_model.py?');
  }
}

init();
