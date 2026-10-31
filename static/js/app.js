import { apiRequest, postJSON } from "./api.js";
import { bindAuth, checkSession, setDashboardLoader, showLanding } from "./auth.js";
import { renderForecast, renderImportance, renderInsights, renderMetricComparison } from "./charts.js";

const $ = (selector) => document.querySelector(selector);
let meta = null;
let toastTimer;
const tabIds = ["forecast", "whatif", "report", "insights", "data"];
const loadedPanels = new Set();

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => element.classList.remove("show"), 4200);
}

function animateCount(element, value, formatter = (number) => Math.round(number).toLocaleString()) {
  const start = performance.now();
  const duration = 800;
  const target = Number(value) || 0;
  function frame(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - (1 - progress) ** 3;
    element.textContent = formatter(target * eased);
    if (progress < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function fillSelect(select, options, labelFor = (value) => value) {
  select.replaceChildren();
  for (const option of options) {
    const element = document.createElement("option");
    element.value = option.value ?? option;
    element.textContent = option.label ?? labelFor(option);
    select.append(element);
  }
}

function fixturesForTeam(team) {
  return (meta?.fixtures || []).filter((fixture) => fixture.team1 === team || fixture.team2 === team);
}

function updateFixtureOptions() {
  const team = $("#scenario-team").value;
  const fixtures = fixturesForTeam(team);
  fillSelect($("#scenario-match"), fixtures.map((fixture) => ({
    value: fixture.date,
    label: `${new Date(`${fixture.date}T12:00:00`).toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" })} · ${fixture.team1 === team ? fixture.team2 : fixture.team1}`,
  })));
}

async function loadMeta() {
  meta = await apiRequest("/api/meta");
  $("#data-badge").textContent = meta.data_source;
  fillSelect($("#forecast-team"), meta.teams);
  fillSelect($("#scenario-team"), meta.teams);
  fillSelect($("#forecast-product"), meta.products);
  fillSelect($("#scenario-product"), meta.products);
  fillSelect($("#record-team"), meta.teams);
  fillSelect($("#record-product"), meta.products);
  const today = new Date();
  $("#record-date").value = [
    today.getFullYear(),
    String(today.getMonth() + 1).padStart(2, "0"),
    String(today.getDate()).padStart(2, "0"),
  ].join("-");
  updateFixtureOptions();
}

async function loadDataStatus() {
  const status = await apiRequest("/api/data/status");
  $("#data-source").textContent = status.source;
  $("#data-record-count").textContent = `${status.record_count} uploaded ${status.record_count === 1 ? "record" : "records"}`;
  const controls = $("#data-controls");
  const message = $("#demo-data-message");
  controls.hidden = status.demo_user;
  message.hidden = !status.demo_user;
  message.textContent = status.demo_user ? status.message : "";
}

async function loadKpis() {
  const kpis = await apiRequest("/api/kpis");
  for (const element of [$("#kpi-units"), $("#kpi-risk"), $("#kpi-uplift")]) {
    element.classList.remove("loading");
  }
  animateCount($("#kpi-units"), kpis.forecast_units_14_days);
  animateCount($("#kpi-risk"), kpis.products_at_stockout_risk);
  animateCount($("#kpi-uplift"), kpis.next_match_uplift_pct, (number) => `${number >= 0 ? "+" : ""}${number.toFixed(1)}%`);
}

async function loadForecast() {
  $("#forecast-loading").hidden = false;
  $("#forecast-chart-wrap").hidden = true;
  try {
    const query = new URLSearchParams({
      team: $("#forecast-team").value,
      product: $("#forecast-product").value,
      days: "14",
    });
    const result = await apiRequest(`/api/forecast?${query}`);
    $("#forecast-chart-wrap").hidden = false;
    renderForecast($("#forecast-chart"), result);
  } finally {
    $("#forecast-loading").hidden = true;
  }
}

async function loadReport() {
  $("#report-loading").hidden = false;
  $("#report-content").hidden = true;
  try {
    const report = await apiRequest("/api/report");
    $("#validation-note").textContent = `Chronological validation holdout beginning ${report.validation_cutoff}. Metrics are read from the saved model evaluation.`;
    const body = $("#metrics-body");
    body.replaceChildren();
    for (const metric of ["MAE", "RMSE", "MAPE"]) {
      const row = document.createElement("tr");
      for (const value of [metric, report.model[metric], report.seasonal_naive[metric]]) {
        const cell = document.createElement(value === metric ? "th" : "td");
        cell.textContent = value === metric ? value : Number(value).toFixed(2);
        row.append(cell);
      }
      body.append(row);
    }
    $("#report-content").hidden = false;
    renderMetricComparison($("#metrics-chart"), report);
    renderImportance($("#importance-chart"), report.feature_importance);
  } finally {
    $("#report-loading").hidden = true;
  }
}

async function loadInsights() {
  $("#insights-loading").hidden = false;
  $("#insights-content").hidden = true;
  try {
    const result = await apiRequest("/api/insights");
    $("#insights-content").hidden = false;
    $("#insights-note").textContent = result.note;
    renderInsights(
      $("#outcome-chart"),
      ["After a win", "After a loss"],
      [result.average_demand_after_wins, result.average_demand_after_losses],
      "Average units",
    );
    renderInsights(
      $("#matchday-chart"),
      ["Match day", "Non-match day"],
      [result.average_match_day_demand, result.average_non_match_day_demand],
      "Average units",
    );
  } finally {
    $("#insights-loading").hidden = true;
  }
}

const panelLoaders = {
  forecast: loadForecast,
  report: loadReport,
  insights: loadInsights,
  data: loadDataStatus,
};

function setTab(tab) {
  for (const id of tabIds) {
    const active = id === tab;
    const button = $(`#tab-${id}`);
    const panel = $(`#panel-${id}`);
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  }
  if (panelLoaders[tab] && !loadedPanels.has(tab)) {
    loadedPanels.add(tab);
    panelLoaders[tab]().catch((error) => toast(error.message));
  } else if (tab === "forecast") {
    loadForecast().catch((error) => toast(error.message));
  }
}

function initializeTabs() {
  for (const id of tabIds) {
    $(`#tab-${id}`).addEventListener("click", () => setTab(id));
  }
}

async function loadDashboard() {
  try {
    await loadMeta();
    const tasks = [loadKpis(), loadForecast(), loadDataStatus()];
    loadedPanels.add("forecast");
    await Promise.all(tasks);
  } catch (error) {
    toast(error.message);
  }
}

async function submitWhatIf(event) {
  event.preventDefault();
  const leadTime = Number($("#lead-time").value);
  if (!Number.isInteger(leadTime) || leadTime < 1 || leadTime > 60) {
    toast("Lead time must be between 1 and 60 days.");
    return;
  }
  $("#whatif-results").hidden = true;
  $("#whatif-loading").hidden = false;
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const outcome = document.querySelector('input[name="outcome"]:checked').value;
    const result = await postJSON("/api/whatif", {
      team: $("#scenario-team").value,
      product: $("#scenario-product").value,
      match_date: $("#scenario-match").value,
      outcome,
      lead_time: leadTime,
    });
    const change = Number(result.demand_change_pct);
    $("#scenario-change").textContent = `${change >= 0 ? "+" : ""}${change.toFixed(1)}%`;
    $("#scenario-duration").textContent = `${result.days_until_uplift_below_5_pct} ${result.days_until_uplift_below_5_pct === 1 ? "day" : "days"}`;
    $("#scenario-stock").textContent = `${Number(result.recommended_stock).toLocaleString()} units`;
    $("#scenario-stock-note").textContent = `Baseline stock: ${Number(result.baseline_stock).toLocaleString()} units`;
    $("#scenario-sentence").textContent = result.sentence;
    $("#win-demand").textContent = `${Number(result.win_demand_7_days).toFixed(1)} units`;
    $("#loss-demand").textContent = `${Number(result.loss_demand_7_days).toFixed(1)} units`;
    $("#whatif-results").hidden = false;
  } catch (error) {
    toast(error.message);
  } finally {
    $("#whatif-loading").hidden = true;
    button.disabled = false;
  }
}

async function refreshPersonalData() {
  await Promise.all([loadDataStatus(), loadKpis(), loadForecast()]);
  loadedPanels.add("forecast");
}

async function uploadSalesData(event) {
  event.preventDefault();
  const file = $("#sales-file").files[0];
  if (!file) {
    toast("Choose a CSV file to upload.");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const result = await apiRequest("/api/data/upload", { method: "POST", body: formData });
    toast(result.message);
    event.currentTarget.reset();
    await refreshPersonalData();
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
}

async function addSalesRow(event) {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const result = await postJSON("/api/data/add-row", {
      date: $("#record-date").value,
      team: $("#record-team").value,
      product: $("#record-product").value,
      units: Number($("#record-units").value),
    });
    toast(result.message);
    $("#record-units").value = "";
    await refreshPersonalData();
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
}

async function resetSalesData() {
  const button = $("#reset-data-button");
  button.disabled = true;
  try {
    const result = await postJSON("/api/data/reset", {});
    toast(result.message);
    await refreshPersonalData();
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
}

function bindDashboard() {
  initializeTabs();
  $("#forecast-team").addEventListener("change", () => loadForecast().catch((error) => toast(error.message)));
  $("#forecast-product").addEventListener("change", () => loadForecast().catch((error) => toast(error.message)));
  $("#scenario-team").addEventListener("change", updateFixtureOptions);
  $("#whatif-form").addEventListener("submit", submitWhatIf);
  $("#upload-form").addEventListener("submit", uploadSalesData);
  $("#add-row-form").addEventListener("submit", addSalesRow);
  $("#reset-data-button").addEventListener("click", resetSalesData);
  $("#logout-button").addEventListener("click", async () => {
    try {
      await postJSON("/api/auth/logout", {});
      loadedPanels.clear();
      showLanding();
    } catch (error) {
      toast(error.message);
    }
  });
  window.addEventListener("fancart:unauthorized", () => {
    loadedPanels.clear();
    showLanding();
    toast("Your session ended. Please sign in again.");
  });
}

bindAuth();
bindDashboard();
setDashboardLoader(loadDashboard);
checkSession(loadDashboard);
