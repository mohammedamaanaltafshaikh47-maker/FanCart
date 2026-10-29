from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from datetime import date, timedelta
from http.cookies import SimpleCookie

import pytest

import app as app_module
from app import REMEMBER_COOKIE, RuntimeState, create_app
from ml.train import DEFAULT_MODEL_DIR


@pytest.fixture(scope="module")
def api_app(tmp_path_factory):
    database = tmp_path_factory.mktemp("fancart-api") / "auth.sqlite3"
    instance = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "DATABASE": database,
            "FANCART_MODEL_DIR": DEFAULT_MODEL_DIR,
            "FANCART_USE_NETWORK": False,
        }
    )
    yield instance


@pytest.fixture
def client(api_app):
    return api_app.test_client()


def test_signup_login_and_logout(client):
    created = client.post(
        "/api/auth/signup",
        json={"username": "api_supporter", "password": "safe-pass"},
    )
    assert created.status_code == 201
    assert created.json["user"]["username"] == "api_supporter"
    assert client.get("/api/auth/me").json["user"]["username"] == "api_supporter"

    client.post("/api/auth/logout")
    signed_in = client.post(
        "/api/auth/login",
        json={"username": "api_supporter", "password": "safe-pass", "remember": False},
    )
    assert signed_in.status_code == 200
    assert signed_in.json["user"]["username"] == "api_supporter"

    signed_out = client.post("/api/auth/logout")
    assert signed_out.status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_protected_routes_require_authentication(client):
    paths = [
        "/api/meta",
        "/api/kpis",
        "/api/forecast?team=Chennai%20Super%20Kings&product=jersey",
        "/api/report",
        "/api/insights",
    ]
    for path in paths:
        response = client.get(path)
        assert response.status_code == 401
        assert response.json == {"error": "Authentication required."}
    response = client.post("/api/whatif", json={})
    assert response.status_code == 401
    assert response.json == {"error": "Authentication required."}


def test_remember_token_authenticates_without_session(client, api_app):
    client.post(
        "/api/auth/signup",
        json={"username": "remember_fan", "password": "safe-pass"},
    )
    client.post("/api/auth/logout")
    response = client.post(
        "/api/auth/login",
        json={"username": "remember_fan", "password": "safe-pass", "remember": True},
    )
    assert response.status_code == 200
    cookie = SimpleCookie()
    cookie.load(response.headers.get("Set-Cookie", ""))
    raw_token = cookie[REMEMBER_COOKIE].value
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    with sqlite3.connect(api_app.config["DATABASE"]) as connection:
        stored = connection.execute(
            "SELECT token_hash FROM tokens WHERE token_hash = ?", (token_hash,)
        ).fetchone()
        plaintext_match = connection.execute(
            "SELECT COUNT(*) FROM tokens WHERE token_hash = ?", (raw_token,)
        ).fetchone()[0]
    assert stored is not None
    assert plaintext_match == 0

    remembered_client = api_app.test_client()
    remembered_client.set_cookie(REMEMBER_COOKIE, raw_token)
    authenticated = remembered_client.get("/api/auth/me")
    assert authenticated.status_code == 200
    assert authenticated.json["user"]["username"] == "remember_fan"


def test_expired_remember_token_is_rejected(client, api_app):
    client.post(
        "/api/auth/signup",
        json={"username": "expired_fan", "password": "safe-pass"},
    )
    client.post("/api/auth/logout")
    response = client.post(
        "/api/auth/login",
        json={"username": "expired_fan", "password": "safe-pass", "remember": True},
    )
    cookie = SimpleCookie()
    cookie.load(response.headers.get("Set-Cookie", ""))
    raw_token = cookie[REMEMBER_COOKIE].value
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    with sqlite3.connect(api_app.config["DATABASE"]) as connection:
        connection.execute(
            "UPDATE tokens SET expires_at = ? WHERE token_hash = ?",
            (time.time() - 1, token_hash),
        )

    expired_client = api_app.test_client()
    expired_client.set_cookie(REMEMBER_COOKIE, raw_token)
    assert expired_client.get("/api/auth/me").status_code == 401


def test_whatif_stock_is_nonnegative_monotonic_and_win_beats_loss(client):
    client.post("/api/auth/demo", json={})
    meta = client.get("/api/meta").json
    fixture = meta["fixtures"][0]
    team = fixture["team1"]
    request_data = {
        "team": team,
        "product": "jersey",
        "match_date": fixture["date"],
        "outcome": "win",
        "lead_time": 2,
    }
    short = client.post("/api/whatif", json=request_data)
    assert short.status_code == 200, short.json
    assert short.json["recommended_stock"] >= 0
    assert short.json["win_demand_7_days"] > short.json["loss_demand_7_days"]

    request_data["lead_time"] = 8
    long = client.post("/api/whatif", json=request_data)
    assert long.status_code == 200, long.json
    assert long.json["recommended_stock"] >= short.json["recommended_stock"]


def test_insights_compare_post_match_demand_and_match_days(client):
    client.post("/api/auth/demo", json={})
    response = client.get("/api/insights")

    assert response.status_code == 200
    assert response.json["average_demand_after_wins"] > response.json["average_demand_after_losses"]
    assert response.json["average_match_day_demand"] > 0
    assert response.json["average_non_match_day_demand"] > 0


def test_health_route_is_public(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_render_cookies_are_secure_httponly_and_lax(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    instance = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "render-test-secret",
            "DATABASE": tmp_path / "render-auth.sqlite3",
            "FANCART_MODEL_DIR": DEFAULT_MODEL_DIR,
        }
    )
    client = instance.test_client()
    signup_response = client.post(
        "/api/auth/signup",
        json={"username": "render_fan", "password": "safe-pass"},
    )
    session_cookie = next(
        value for value in signup_response.headers.getlist("Set-Cookie")
        if value.startswith("session=")
    )
    assert "Secure" in session_cookie
    assert "HttpOnly" in session_cookie
    assert "SameSite=Lax" in session_cookie

    client.post("/api/auth/logout")
    login_response = client.post(
        "/api/auth/login",
        json={"username": "render_fan", "password": "safe-pass", "remember": True},
    )
    remember_cookie = next(
        value for value in login_response.headers.getlist("Set-Cookie")
        if value.startswith(f"{REMEMBER_COOKIE}=")
    )
    assert "Secure" in remember_cookie
    assert "HttpOnly" in remember_cookie
    assert "SameSite=Lax" in remember_cookie


def test_missing_model_trains_in_background_and_api_reports_status(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked_build(*_args, **_kwargs):
        started.set()
        if not release.wait(timeout=10):
            raise TimeoutError("Test background model startup timed out.")
        finished.set()
        return RuntimeState(
            matches=None,
            fixtures=None,
            sales=None,
            source="generated sample",
            model={},
        )

    monkeypatch.setattr(app_module, "_build_runtime", blocked_build)
    instance = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "background-test-secret",
            "DATABASE": tmp_path / "background-auth.sqlite3",
            "FANCART_MODEL_DIR": tmp_path / "missing-model",
            "FANCART_USE_NETWORK": False,
        }
    )
    client = instance.test_client()
    try:
        assert started.wait(timeout=2)
        client.post("/api/auth/demo", json={})
        response = client.get("/api/meta")
        assert response.status_code == 503
        assert response.json == {"error": "The model is training. Please try again shortly."}
        assert client.get("/healthz").json == {"status": "ok"}
    finally:
        release.set()
    assert finished.wait(timeout=2)


def test_whatif_rejects_malformed_outcome(client):
    client.post("/api/auth/demo", json={})
    response = client.post(
        "/api/whatif",
        json={
            "team": "Chennai Super Kings",
            "product": "jersey",
            "match_date": (date.today() + timedelta(days=1)).isoformat(),
            "outcome": [],
            "lead_time": 7,
        },
    )

    assert response.status_code == 400
    assert response.json == {"error": "Outcome must be win or loss."}
