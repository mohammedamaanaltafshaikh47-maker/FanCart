const green = "#00ff87";
const blue = "#66a8ff";
const muted = "#9aa8c2";
const grid = "rgba(154, 168, 194, .13)";
const activeCharts = new Map();

function replaceChart(canvas, config) {
  const prior = activeCharts.get(canvas.id);
  if (prior) prior.destroy();
  if (window.Chart) {
    activeCharts.set(canvas.id, new window.Chart(canvas, config));
  } else {
    drawFallback(canvas, config);
  }
}

function drawFallback(canvas, config) {
  const context = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  const datasets = config.data.datasets || [];
  const values = datasets.flatMap((dataset) => dataset.data.map((point) => Number(point.y ?? point)));
  if (!values.length) return;
  const max = Math.max(...values) || 1;
  context.strokeStyle = grid;
  context.fillStyle = muted;
  context.font = "11px sans-serif";
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = 16 + (height - 44) * tick / 4;
    context.beginPath();
    context.moveTo(42, y);
    context.lineTo(width - 8, y);
    context.stroke();
  }
  const first = datasets[0];
  context.strokeStyle = first.borderColor || green;
  context.lineWidth = 2;
  context.beginPath();
  first.data.forEach((value, index) => {
    const number = Number(value.y ?? value);
    const x = 44 + (width - 54) * index / Math.max(first.data.length - 1, 1);
    const y = height - 27 - (height - 54) * number / max;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

const axisOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { intersect: false, mode: "index" },
  plugins: {
    legend: { labels: { color: muted, usePointStyle: true, boxWidth: 8 } },
    tooltip: { backgroundColor: "#0a1128", titleColor: "#f0f5ff", bodyColor: "#dce6f6", borderColor: "#344260", borderWidth: 1 },
  },
  scales: {
    x: { ticks: { color: muted, maxTicksLimit: 9 }, grid: { color: grid } },
    y: { beginAtZero: true, ticks: { color: muted }, grid: { color: grid } },
  },
};

export function renderForecast(canvas, response) {
  const labels = [
    ...response.history.map((item) => item.date),
    ...response.forecast.map((item) => item.date),
  ];
  const actual = [
    ...response.history.map((item) => item.units),
    ...response.forecast.map(() => null),
  ];
  const low = [...response.history.map(() => null), ...response.forecast.map((item) => item.lower)];
  const upper = [...response.history.map(() => null), ...response.forecast.map((item) => item.upper)];
  const median = [...response.history.map(() => null), ...response.forecast.map((item) => item.median)];
  replaceChart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Actual", data: actual, borderColor: blue, backgroundColor: blue, pointRadius: 0, borderWidth: 2, spanGaps: false },
        { label: "10th percentile", data: low, borderColor: "transparent", pointRadius: 0, borderWidth: 0 },
        { label: "10–90th percentile range", data: upper, borderColor: "transparent", backgroundColor: "rgba(0,255,135,.13)", pointRadius: 0, fill: "-1", borderWidth: 0 },
        { label: "Forecast", data: median, borderColor: green, backgroundColor: green, pointRadius: 0, borderWidth: 2, borderDash: [6, 4] },
      ],
    },
    options: { ...axisOptions, scales: { ...axisOptions.scales, x: { ...axisOptions.scales.x, ticks: { ...axisOptions.scales.x.ticks, maxTicksLimit: 12 } } } },
  });
}

export function renderMetricComparison(canvas, report) {
  const keys = ["MAE", "RMSE", "MAPE"];
  replaceChart(canvas, {
    type: "bar",
    data: {
      labels: keys,
      datasets: [
        { label: "FanCart model", data: keys.map((key) => report.model[key]), backgroundColor: green, borderRadius: 5 },
        { label: "Seasonal naive", data: keys.map((key) => report.seasonal_naive[key]), backgroundColor: blue, borderRadius: 5 },
      ],
    },
    options: { ...axisOptions, scales: { ...axisOptions.scales, x: { ...axisOptions.scales.x, grid: { display: false } } } },
  });
}

export function renderImportance(canvas, values) {
  const ordered = values.slice(0, 9);
  replaceChart(canvas, {
    type: "bar",
    data: { labels: ordered.map((item) => item.feature), datasets: [{ label: "Importance", data: ordered.map((item) => item.importance), backgroundColor: green, borderRadius: 4 }] },
    options: {
      ...axisOptions,
      indexAxis: "y",
      plugins: { ...axisOptions.plugins, legend: { display: false } },
      scales: { x: { beginAtZero: true, ticks: { color: muted }, grid: { color: grid } }, y: { ticks: { color: muted }, grid: { display: false } } },
    },
  });
}

export function renderInsights(canvas, labels, values, label) {
  replaceChart(canvas, {
    type: "bar",
    data: { labels, datasets: [{ label, data: values, backgroundColor: [green, blue], borderRadius: 7 }] },
    options: { ...axisOptions, plugins: { ...axisOptions.plugins, legend: { display: false } }, scales: { ...axisOptions.scales, x: { ...axisOptions.scales.x, grid: { display: false } } } },
  });
}
