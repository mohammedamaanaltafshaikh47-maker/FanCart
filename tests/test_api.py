from __future__ import annotations

import hashlib
import io
import sqlite3
import threading
import time
from datetime import date, timedelta
from http.cookies import SimpleCookie

import pytest

import app as app_module
from app import REMEMBER_COOKIE, RuntimeState, create_app
from data.synthetic import generate_matches
from ml.train import DEFAULT_MODEL_DIR
from scripts import export_sample_csv


@pytest.fixture(scope="module")
def api_app(tmp_path_factory):
    database = tmp_path_factory.mktemp("fancart-api") / "auth.sqlite3"
    user_models = tmp_path_factory.mktemp("fancart-user-models")
    instance = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "DATABASE": database,
            "FANCART_MODEL_DIR": DEFAULT_MODEL_DIR,
            "FANCART_USER_MODEL_DIR": user_models,
            "FANCART_USE_NETWORK": False,
        }
    )
    yield instance


@pytest.fixture
def client(api_app):
    return api_app.test_client()


def sales_csv(days=90, team="Sample United", product="scarf", units=100):
    first_day = date.today() - timedelta(days=days - 1)
    lines = ["date,team,product,units,price"]
    for offset in range(days):
        lines.append(
            f"{(first_day + timedelta(days=offset)).isoformat()},{team},{product},{units},24.5"
        )
    return ("\n".join(lines) + "\n").encode()


def upload_csv(client, content):
    return client.post(
        "/api/data/upload",
        data={"file": (io.BytesIO(content), "sales.csv")},
        content_type="multipart/form-data",
    )


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
        "/api/data/status",
        "/api/data/sample",
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


def test_demo_user_cannot_modify_sample_data(client):
    client.post("/api/auth/demo", json={})
    expected_error = {
        "error": "Demo user uses the sample data. Create an account to use your own data."
    }

    upload = client.post(
        "/api/data/upload",
        data={
            "file": (
                io.BytesIO(b"date,team,product,units\n2026-10-28,Chennai Super Kings,jersey,999\n"),
                "sales.csv",
            )
        },
        content_type="multipart/form-data",
    )
    add_row = client.post(
        "/api/data/add-row",
        json={
            "date": date.today().isoformat(),
            "team": "Chennai Super Kings",
            "product": "jersey",
            "units": 999,
        },
    )
    reset = client.post("/api/data/reset")

    for response in (upload, add_row, reset):
        assert response.status_code == 403
        assert response.json == expected_error
    status = client.get("/api/data/status")
    assert status.json["source"] == "synthetic"
    assert status.json["message"] == expected_error["error"]


def test_demo_kpis_are_unchanged_after_another_user_uploads(api_app):
    demo_client = api_app.test_client()
    user_client = api_app.test_client()
    demo_client.post("/api/auth/demo", json={})
    before = demo_client.get(
        "/api/forecast?team=Chennai%20Super%20Kings&product=jersey"
    )
    assert before.status_code == 200

    signup = user_client.post(
        "/api/auth/signup",
        json={"username": "isolated_data_fan", "password": "safe-pass"},
    )
    assert signup.status_code == 201
    upload = upload_csv(user_client, sales_csv(units=100000))
    assert upload.status_code == 200, upload.json
    assert user_client.get("/api/data/status").json["source"] == "uploaded"

    after = demo_client.get(
        "/api/forecast?team=Chennai%20Super%20Kings&product=jersey"
    )
    assert after.status_code == 200
    assert after.json == before.json


def test_valid_upload_drives_user_meta_and_forecast(api_app):
    user = api_app.test_client()
    assert user.post(
        "/api/auth/signup",
        json={"username": "valid_upload_fan", "password": "safe-pass"},
    ).status_code == 201

    uploaded = upload_csv(user, sales_csv())
    assert uploaded.status_code == 200, uploaded.json
    status = user.get("/api/data/status").json
    assert status["source"] == "uploaded"
    assert status["rows"] == 90
    assert status["teams"] == 1
    assert status["products"] == 1
    assert status["date_min"] == (date.today() - timedelta(days=89)).isoformat()
    assert status["date_max"] == date.today().isoformat()

    meta = user.get("/api/meta").json
    assert meta["teams"] == ["Sample United"]
    assert meta["products"] == ["scarf"]
    assert meta["sales_note"] == "Using your uploaded data."
    forecast = user.get("/api/forecast?team=Sample%20United&product=scarf")
    assert forecast.status_code == 200, forecast.json
    assert len(forecast.json["history"]) == 90
    assert len(forecast.json["forecast"]) == 14


def test_exported_sample_csv_passes_upload_validation(api_app, tmp_path, monkeypatch):
    monkeypatch.setattr(
        export_sample_csv,
        "load_match_data",
        lambda: (generate_matches(), "generated sample"),
    )
    csv_path = export_sample_csv.export_sample_csv(tmp_path / "sample_sales_data.csv")
    assert csv_path.stat().st_size <= 2 * 1024 * 1024
    user = api_app.test_client()
    assert user.post(
        "/api/auth/signup",
        json={"username": "export_csv_fan", "password": "safe-pass"},
    ).status_code == 201

    uploaded = upload_csv(user, csv_path.read_bytes())
    assert uploaded.status_code == 200, uploaded.json
    assert uploaded.json["record_count"] == 6 * 3 * 731
    status = user.get("/api/data/status").json
    assert status["source"] == "uploaded"
    assert status["teams"] == 6
    assert status["products"] == 3
    assert status["date_min"] == (date.today() - timedelta(days=730)).isoformat()
    assert status["date_max"] == date.today().isoformat()


def test_sample_csv_download_requires_login_and_returns_attachment(client):
    signup = client.post(
        "/api/auth/signup",
        json={"username": "sample_download_fan", "password": "safe-pass"},
    )
    assert signup.status_code == 201

    response = client.get("/api/data/sample")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "sample_sales_data.csv" in response.headers["Content-Disposition"]
    assert response.get_data().startswith(b"date,team,product,units,price")


def test_upload_rejects_missing_column(client):
    client.post(
        "/api/auth/signup",
        json={"username": "missing_column_fan", "password": "safe-pass"},
    )
    content = sales_csv().decode().splitlines()
    response = upload_csv(client, "\n".join(line.rsplit(",", 1)[0] for line in content).encode())

    assert response.status_code == 400
    assert response.json == {"error": "CSV is missing required column(s): price."}


def test_upload_rejects_negative_units_with_row_number(client):
    client.post(
        "/api/auth/signup",
        json={"username": "negative_units_fan", "password": "safe-pass"},
    )
    content = sales_csv().decode().replace(",100,24.5", ",-1,24.5", 1).encode()
    response = upload_csv(client, content)

    assert response.status_code == 400
    assert response.json == {"error": "Row 2: units must be at least 0."}


def test_upload_rejects_bad_date_with_row_number(client):
    client.post(
        "/api/auth/signup",
        json={"username": "bad_date_fan", "password": "safe-pass"},
    )
    lines = sales_csv().decode().splitlines()
    lines[1] = "not-a-date" + lines[1][10:]
    response = upload_csv(client, "\n".join(lines).encode())

    assert response.status_code == 400
    assert response.json == {"error": "Row 2: date is invalid."}


def test_upload_rejects_too_little_history(client):
    client.post(
        "/api/auth/signup",
        json={"username": "short_history_fan", "password": "safe-pass"},
    )
    response = upload_csv(client, sales_csv(days=30))

    assert response.status_code == 400
    assert "at least 90 days of history" in response.json["error"]


def test_add_row_and_reset_return_to_sample_data(client):
    client.post(
        "/api/auth/signup",
        json={"username": "add_reset_fan", "password": "safe-pass"},
    )
    added = client.post(
        "/api/data/add-row",
        json={
            "date": date.today().isoformat(),
            "team": "Local Club",
            "product": "scarf",
            "units": 12.5,
            "price": 19.95,
        },
    )
    assert added.status_code == 201, added.json
    status = client.get("/api/data/status").json
    assert status["source"] == "uploaded"
    assert status["rows"] == 1

    reset = client.post("/api/data/reset")
    assert reset.status_code == 200
    status = client.get("/api/data/status").json
    assert status["source"] == "synthetic"
    assert status["rows"] > 1


def test_new_account_starts_with_synthetic_data(client):
    signup = client.post(
        "/api/auth/signup",
        json={"username": "new_synthetic_fan", "password": "safe-pass"},
    )
    assert signup.status_code == 201

    status = client.get("/api/data/status")
    assert status.status_code == 200
    assert status.json["source"] == "synthetic"
    assert status.json["rows"] > 0
    assert status.json["date_min"]
    assert status.json["date_max"]
    assert status.json["demo_user"] is False


def test_demo_user_can_start_with_empty_database(tmp_path):
    instance = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "empty-demo-test-secret",
            "DATABASE": tmp_path / "empty-demo.sqlite3",
            "FANCART_MODEL_DIR": DEFAULT_MODEL_DIR,
            "FANCART_USE_NETWORK": False,
        }
    )
    client = instance.test_client()

    demo = client.post("/api/auth/demo", json={})
    assert demo.status_code == 200
    assert demo.json["user"]["username"] == "demo_fan"
    assert client.get("/api/auth/me").json["user"]["username"] == "demo_fan"
    assert client.get("/api/data/status").json["source"] == "synthetic"


def test_uploaded_sales_are_private_to_the_account(api_app):
    owner = api_app.test_client()
    other_user = api_app.test_client()
    assert owner.post(
        "/api/auth/signup",
        json={"username": "private_data_owner", "password": "safe-pass"},
    ).status_code == 201
    assert other_user.post(
        "/api/auth/signup",
        json={"username": "separate_data_user", "password": "safe-pass"},
    ).status_code == 201

    uploaded = owner.post(
        "/api/data/add-row",
        json={
            "date": date.today().isoformat(),
            "team": "Chennai Super Kings",
            "product": "jersey",
            "units": 1234,
            "price": 42.0,
        },
    )
    assert uploaded.status_code == 201
    assert owner.get("/api/data/status").json["source"] == "uploaded"
    assert other_user.get("/api/data/status").json["source"] == "synthetic"
    assert other_user.get("/api/data/status").json["rows"] > 0


def test_demo_username_cannot_be_registered(client):
    response = client.post(
        "/api/auth/signup",
        json={"username": "DEMO_FAN", "password": "safe-pass"},
    )

    assert response.status_code == 409
    assert response.json == {"error": "That username is reserved."}


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
