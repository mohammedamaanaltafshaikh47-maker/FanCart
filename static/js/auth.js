import { apiRequest, postJSON } from "./api.js";

const landing = document.querySelector("#landing");
const dashboard = document.querySelector("#dashboard");
const form = document.querySelector("#auth-form");
const errorLine = document.querySelector("#auth-error");
const title = document.querySelector("#auth-title");
const submit = document.querySelector("#auth-submit");
const password = document.querySelector("#password");
const remember = document.querySelector("#remember");
const loginMode = document.querySelector("#login-mode");
const signupMode = document.querySelector("#signup-mode");
const switchControl = document.querySelector(".auth-switch");
let mode = "login";

function showAuthError(message) {
  errorLine.textContent = message;
}

function setMode(nextMode) {
  if (mode === nextMode) return;
  mode = nextMode;
  form.classList.add("switching");
  switchControl.classList.toggle("signup", mode === "signup");
  loginMode.classList.toggle("active", mode === "login");
  signupMode.classList.toggle("active", mode === "signup");
  loginMode.setAttribute("aria-selected", String(mode === "login"));
  signupMode.setAttribute("aria-selected", String(mode === "signup"));
  showAuthError("");
  window.setTimeout(() => {
    const creating = mode === "signup";
    title.textContent = creating ? "Build your stock edge." : "Demand that backs your team.";
    submit.textContent = creating ? "Create account" : "Log in";
    password.autocomplete = creating ? "new-password" : "current-password";
    remember.closest("label").hidden = creating;
    form.classList.remove("switching");
  }, 130);
}

export function showDashboard(user, onReady) {
  document.querySelector("#username-label").textContent = user.username;
  document.querySelector("#welcome-message").textContent = `Welcome back, ${user.username}. Your next forecast is ready.`;
  landing.classList.add("leaving");
  window.setTimeout(() => {
    landing.hidden = true;
    dashboard.hidden = false;
    requestAnimationFrame(() => dashboard.classList.add("is-visible"));
    onReady();
  }, 430);
}

export function showLanding() {
  dashboard.classList.remove("is-visible");
  dashboard.hidden = true;
  landing.hidden = false;
  landing.classList.remove("leaving");
  document.querySelector("#username").focus();
}

async function submitAuth(event) {
  event.preventDefault();
  showAuthError("");
  const username = document.querySelector("#username").value.trim();
  const secret = password.value;
  if (username.length < 3 || (mode === "signup" && username.length > 64)) {
    showAuthError("Use a username between 3 and 64 characters.");
    return;
  }
  if (secret.length < 6 || secret.length > 256) {
    showAuthError("Your password must be between 6 and 256 characters.");
    return;
  }
  submit.disabled = true;
  try {
    const endpoint = mode === "login" ? "/api/auth/login" : "/api/auth/signup";
    const payload = await postJSON(
      endpoint,
      { username, password: secret, remember: remember.checked },
      { allowUnauthorized: true },
    );
    showDashboard(payload.user, onReady);
  } catch (error) {
    showAuthError(error.message);
  } finally {
    submit.disabled = false;
  }
}

export async function startDemo() {
  const button = document.querySelector("#demo-button");
  button.disabled = true;
  try {
    const payload = await postJSON("/api/auth/demo", {}, { allowUnauthorized: true });
    showDashboard(payload.user, onReady);
  } catch (error) {
    showAuthError(error.message);
  } finally {
    button.disabled = false;
  }
}

let onReady = () => {};
export function setDashboardLoader(loader) {
  onReady = loader;
}

export async function checkSession(onAuthenticated) {
  onReady = onAuthenticated;
  try {
    const payload = await apiRequest("/api/auth/me", { allowUnauthorized: true });
    showDashboard(payload.user, onAuthenticated);
  } catch (error) {
    if (error.status !== 401) {
      showAuthError(error.message);
    }
    showLanding();
  }
}

export function bindAuth() {
  loginMode.addEventListener("click", () => setMode("login"));
  signupMode.addEventListener("click", () => setMode("signup"));
  form.addEventListener("submit", submitAuth);
  document.querySelector("#demo-button").addEventListener("click", startDemo);
}
