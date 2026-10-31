import { apiRequest, postJSON } from "./api.js";
import { bindAuth, checkSession, setDashboardLoader, showLanding } from "./auth.js";
import { renderForecast, renderImportance, renderInsights, renderMetricComparison } from "./charts.js";

const $ = (selector) => document.querySelector(selector);
let meta = null;
let toastTimer;
let trainingPollTimer;
let latestDataStatus = null;
const tabIds = ["forecast", "whatif", "report", "insights"];
const loadedPanels = new Set();

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => element.classList.remove("show"), 4200);
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
  const previous = {
    forecastTeam: $("#forecast-team").value,
    forecastProduct: $("#forecast-product").value,
    scenarioTeam: $("#scenario-team").value,
    scenarioProduct: $("#scenario-product").value,
    recordTeam: $("#record-team").value,
    recordProduct: $("#record-product").value,
  };
  meta = await apiRequest("/api/meta");
  $("#data-badge").textContent = meta.data_source;
  for (const [selector, value, options] of [
    ["#forecast-team", previous.forecastTeam, meta.teams],
    ["#scenario-team", previous.scenarioTeam, meta.teams],
    ["#forecast-product", previous.forecastProduct, meta.products],
    ["#scenario-product", previous.scenarioProduct, meta.products],
    ["#record-team", previous.recordTeam, meta.teams],
    ["#record-product", previous.recordProduct, meta.products],
  ]) {
    fillSelect($(selector), options);
    if (options.includes(value)) $(selector).value = value;
  }
  $("#sales-source-note").textContent = meta.sales_note === "Using your uploaded data."
    ? "Using your uploaded data. The forecast includes a 10th–90th percentile range."
    : "The solid line shows synthetic sales history; the forecast includes a 10th–90th percentile range.";
  $("#dashboard-footer").textContent = meta.sales_note === "Using your uploaded data."
    ? "FanCart is a planning aid. Match schedules are simulated. Using your uploaded data."
    : "FanCart is a planning aid. Match schedules are simulated; sales data is synthetic.";
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
  latestDataStatus = status;
  $("#data-source").textContent = status.source === "uploaded"
    ? "Your uploaded data"
    : "Sample data (synthetic)";
  $("#data-rows").textContent = Number(status.rows).toLocaleString();
  $("#data-date-range").textContent = status.date_min && status.date_max
    ? `${status.date_min} – ${status.date_max}`
    : "—";
  $("#data-teams").textContent = Number(status.teams).toLocaleString();
  $("#data-products").textContent = Number(status.products).toLocaleString();
  const modelStatus = $("#model-status");
  modelStatus.textContent = status.training ? "Training your model..." : "Ready";
  modelStatus.classList.toggle("is-training", status.training);
  modelStatus.title = status.last_error || "";
  const modelError = $("#data-model-error");
  modelError.hidden = !status.last_error;
  modelError.textContent = status.last_error ? `Model training failed: ${status.last_error}` : "";
  const controls = $("#data-controls");
  const message = $("#demo-data-message");
  controls.disabled = status.demo_user;
  message.hidden = !status.demo_user;
  message.textContent = status.demo_user ? status.message : "";
  if (status.training && !trainingPollTimer) {
    trainingPollTimer = window.setTimeout(async () => {
      trainingPollTimer = null;
      try {
        const wasTraining = latestDataStatus?.training;
        await loadDataStatus();
        if (wasTraining && !latestDataStatus?.training) {
          if (latestDataStatus.last_error) toast(latestDataStatus.last_error);
          await refreshAfterTraining();
        }
      } catch (error) {
        toast(error.message);
      }
    }, 2000);
  } else if (!status.training && trainingPollTimer) {
    window.clearTimeout(trainingPollTimer);
    trainingPollTimer = null;
  }
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
    $("#report-sales-note").textContent = report.note === "Using your uploaded data."
      ? "Using your uploaded data. Validation metrics reflect your trained model."
      : "Sales data is synthetic. Validation scores should not be interpreted as measured real-world store accuracy.";
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
    const tasks = [loadForecast(), loadDataStatus()];
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
  await Promise.all([loadMeta(), loadDataStatus()]);
  await loadForecast();
  loadedPanels.add("forecast");
}

async function refreshAfterTraining() {
  await loadMeta();
  await loadForecast();
  const refreshes = [];
  if (loadedPanels.has("report")) refreshes.push(loadReport());
  if (loadedPanels.has("insights")) refreshes.push(loadInsights());
  $("#whatif-results").hidden = true;
  await Promise.all(refreshes);
}

async function uploadSalesData(event) {
  event.preventDefault();
  const form = event.currentTarget;
  if (!form || typeof form.querySelector !== "function") {
    toast("The upload form is unavailable. Reload the page and try again.");
    return;
  }
  const fileInput = $("#sales-file");
  const file = fileInput?.files?.[0];
  if (!file) {
    toast("Choose a CSV file to upload.");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  const button = form.querySelector("button[type=submit]");
  if (button) button.disabled = true;
  try {
    const result = await apiRequest("/api/data/upload", { method: "POST", body: formData });
    toast(result.message);
    if (fileInput) fileInput.value = "";
    const fileLabel = $("#upload-file-label");
    if (fileLabel) fileLabel.textContent = "Drop a CSV here or browse";
    await refreshPersonalData();
  } catch (error) {
    toast(error.message);
  } finally {
    if (button) button.disabled = false;
  }
}

async function addSalesRow(event) {
  event.preventDefault();
  const form = event.currentTarget;
  if (!form || typeof form.querySelector !== "function") {
    toast("The record form is unavailable. Reload the page and try again.");
    return;
  }
  const dateInput = $("#record-date");
  const teamSelect = $("#record-team");
  const productSelect = $("#record-product");
  const unitsInput = $("#record-units");
  const priceInput = $("#record-price");
  if (!dateInput || !teamSelect || !productSelect || !unitsInput || !priceInput) {
    toast("The record form is incomplete. Reload the page and try again.");
    return;
  }
  const button = form.querySelector("button[type=submit]");
  if (button) button.disabled = true;
  try {
    const result = await postJSON("/api/data/add-row", {
      date: dateInput.value,
      team: teamSelect.value,
      product: productSelect.value,
      units: Number(unitsInput.value),
      price: Number(priceInput.value),
    });
    toast(result.message);
    $("#record-dialog")?.close();
    unitsInput.value = "";
    priceInput.value = "";
    await refreshPersonalData();
  } catch (error) {
    toast(error.message);
  } finally {
    if (button) button.disabled = false;
  }
}

async function resetSalesData() {
  const button = $("#reset-data-button");
  if (!button) {
    toast("The reset control is unavailable. Reload the page and try again.");
    return;
  }
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
  $("#open-record-dialog").addEventListener("click", () => $("#record-dialog").showModal());
  $("#close-record-dialog").addEventListener("click", () => $("#record-dialog").close());
  $("#reset-data-button").addEventListener("click", resetSalesData);
  $("#sales-file").addEventListener("change", (event) => {
    $("#upload-file-label").textContent = event.currentTarget.files[0]?.name || "Drop a CSV here or browse";
  });
  const dropZone = $("#upload-drop-zone");
  for (const eventName of ["dragenter", "dragover"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("is-dragging");
    });
  }
  dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      $("#sales-file").files = transfer.files;
      $("#upload-file-label").textContent = file.name;
    }
  });
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
