from __future__ import annotations

import json
import io
import logging
import os
import re
import secrets
import threading
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, session
from sklearn.inspection import permutation_importance

from auth import db
from auth.service import (
    TOKEN_LIFETIME_SECONDS,
    ensure_demo_user,
    login,
    revoke_token,
    signup,
    validate_token,
)
from data.loader import load_match_data
from data.synthetic import ACTIVE_TEAMS, PRODUCTS, generate_fixtures, generate_sales
from data import user_data
from ml.features import FEATURE_COLUMNS, build_features
from ml.predict import (
    compare_scenarios,
    forecast_product,
    forecast_products_batch,
    recommend_stock,
)
from ml.train import DEFAULT_MODEL_DIR, load_or_train

APP_ROOT = Path(__file__).resolve().parent
REMEMBER_COOKIE = "fancart_remember"
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")


@dataclass
class RuntimeState:
    matches: pd.DataFrame
    fixtures: pd.DataFrame
    sales: pd.DataFrame
    source: str
    model: dict[str, Any]


class ModelUnavailableError(Exception):
    pass


def _secret_key() -> str:
    return os.environ.get("SECRET_KEY") or secrets.token_urlsafe(48)


def _build_runtime(model_dir: Path, use_network: bool, announce: bool) -> RuntimeState:
    model_path = model_dir / "fancart_model.joblib"
    if announce:
        print("Loading match data (Cricsheet with generated-data fallback)...", flush=True)
    if use_network:
        matches, source = load_match_data()
    else:
        from data.synthetic import generate_matches

        matches, source = generate_matches(), "generated sample"
        matches["simulated"] = True
    today = date.today()
    fixtures = generate_fixtures(ACTIVE_TEAMS, today, days=30)
    sales = generate_sales(
        matches,
        start=today - timedelta(days=730),
        end=today,
    )
    features = build_features(sales, matches, fixtures)
    if announce:
        if model_path.exists():
            print("Loading the saved FanCart model...", flush=True)
        else:
            print("No saved model found; training the demand model...", flush=True)
    model = load_or_train(features, model_dir)
    if announce:
        print(f"FanCart model ready. Match data: {source}.", flush=True)
    return RuntimeState(matches, fixtures, sales, source, model)


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    is_render = "RENDER" in os.environ
    app.config.update(
        SECRET_KEY=_secret_key(),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=is_render,
        REMEMBER_COOKIE_SECURE=is_render,
        FANCART_MODEL_DIR=DEFAULT_MODEL_DIR,
        FANCART_USE_NETWORK=True,
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    )
    if config:
        app.config.update(config)
    db.initialize(app.config.get("DATABASE"))
    user_data.initialize(app.config.get("DATABASE"))
    app.extensions["fancart_runtime"] = app.config.get("FANCART_RUNTIME")
    runtime_lock = threading.Lock()
    app.extensions["fancart_runtime_lock"] = runtime_lock
    training_status = {"started": False, "error": None}
    app.extensions["fancart_training_status"] = training_status

    def train_runtime_in_background() -> None:
        try:
            state = _build_runtime(
                Path(app.config["FANCART_MODEL_DIR"]),
                bool(app.config["FANCART_USE_NETWORK"]),
                bool(app.config.get("FANCART_ANNOUNCE_STARTUP", False)),
            )
        except Exception as error:
            app.logger.exception("Background model training or data initialization failed.")
            with runtime_lock:
                training_status["error"] = str(error)
            return
        with runtime_lock:
            app.extensions["fancart_runtime"] = state

    def runtime() -> RuntimeState:
        state = app.extensions.get("fancart_runtime")
        if state is not None:
            return state
        model_path = Path(app.config["FANCART_MODEL_DIR"]) / "fancart_model.joblib"
        if not model_path.is_file():
            with runtime_lock:
                error = training_status["error"]
                if error:
                    raise ModelUnavailableError(
                        f"Model training failed: {error}"
                    )
                if not training_status["started"]:
                    training_status["started"] = True
                    threading.Thread(
                        target=train_runtime_in_background,
                        name="fancart-model-training",
                        daemon=True,
                    ).start()
            raise ModelUnavailableError("The model is training. Please try again shortly.")
        with app.extensions["fancart_runtime_lock"]:
            state = app.extensions.get("fancart_runtime")
            if state is None:
                try:
                    state = _build_runtime(
                        Path(app.config["FANCART_MODEL_DIR"]),
                        bool(app.config["FANCART_USE_NETWORK"]),
                        bool(app.config.get("FANCART_ANNOUNCE_STARTUP", False)),
                    )
                except Exception:
                    app.logger.exception("Unable to initialize FanCart data and model.")
                    raise
                app.extensions["fancart_runtime"] = state
        return state

    app.extensions["fancart_initialize_runtime"] = runtime

    def api_error(message: str, status: int):
        return jsonify({"error": message}), status

    def json_body() -> dict[str, Any] | None:
        body = request.get_json(silent=True)
        return body if isinstance(body, dict) else None

    def authenticated_username() -> str | None:
        username = session.get("username")
        if isinstance(username, str):
            with db.connect(app.config.get("DATABASE")) as connection:
                user = connection.execute(
                    "SELECT username FROM users WHERE username = ? COLLATE NOCASE",
                    (username,),
                ).fetchone()
            if user is not None:
                return str(user["username"])
            session.clear()
        token = request.cookies.get(REMEMBER_COOKIE, "")
        return validate_token(token, app.config.get("DATABASE")) if token else None

    def is_demo_user(username: str | None = None) -> bool:
        current_username = username if username is not None else authenticated_username()
        return current_username is not None and current_username.casefold() == "demo_fan"

    def current_data_user_id() -> int:
        username = authenticated_username()
        user_id = user_data.user_id(username or "", app.config.get("DATABASE"))
        if user_id is None:
            raise RuntimeError("The authenticated user no longer exists.")
        return user_id

    def current_runtime() -> RuntimeState:
        base = runtime()
        username = authenticated_username()
        if username is None or is_demo_user(username):
            return base
        rows = user_data.records(current_data_user_id(), app.config.get("DATABASE"))
        if not rows:
            return base
        overrides = pd.DataFrame(rows)
        overrides["date"] = pd.to_datetime(overrides["date"]).dt.normalize()
        overrides["units"] = overrides["units"].astype(int)
        sales = base.sales.copy()
        sales["date"] = pd.to_datetime(sales["date"]).dt.normalize()
        keys = ["date", "team", "product"]
        replacements = overrides.set_index(keys)["units"]
        sales_index = pd.MultiIndex.from_frame(sales[keys])
        custom_index = pd.MultiIndex.from_frame(overrides[keys])
        replaced = sales_index.isin(custom_index)
        sales.loc[replaced, "units"] = replacements.reindex(sales_index[replaced]).to_numpy()
        missing_custom = ~custom_index.isin(sales_index)
        if missing_custom.any():
            additional = overrides.loc[missing_custom].copy()
            additional["price"] = additional["product"].map(
                {"jersey": 42.0, "cap": 18.0, "flag": 12.0}
            )
            sales = pd.concat([sales, additional], ignore_index=True)
        sales = sales.sort_values(["team", "product", "date"]).reset_index(drop=True)
        return RuntimeState(base.matches, base.fixtures, sales, base.source, base.model)

    def validated_sales_row(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("Each record must be a JSON object.")
        try:
            row_date = date.fromisoformat(str(raw.get("date", ""))).isoformat()
        except ValueError as error:
            raise ValueError("Date must use YYYY-MM-DD format.") from error
        if date.fromisoformat(row_date) > date.today():
            raise ValueError("Sales records cannot be dated in the future.")
        team, product = raw.get("team"), raw.get("product")
        if team not in ACTIVE_TEAMS:
            raise ValueError("Select a valid team.")
        if product not in PRODUCTS:
            raise ValueError("Select a valid product.")
        units = raw.get("units")
        if isinstance(units, bool) or not isinstance(units, (int, float)):
            raise ValueError("Units must be a non-negative whole number.")
        if not np.isfinite(units) or int(units) != units or not 0 <= units <= 1_000_000:
            raise ValueError("Units must be a non-negative whole number up to 1000000.")
        return {"date": row_date, "team": team, "product": product, "units": int(units)}

    demo_data_error = {
        "error": "Demo user uses the sample data. Create an account to use your own data."
    }

    @app.before_request
    def require_authentication():
        if not request.path.startswith("/api/"):
            return None
        public = {
            "/api/auth/signup",
            "/api/auth/login",
            "/api/auth/demo",
            "/api/auth/logout",
            "/api/auth/me",
        }
        if request.path in public:
            return None
        if authenticated_username() is None:
            return api_error("Authentication required.", 401)
        return None

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return api_error("API endpoint not found.", 404)
        return "Not found", 404

    @app.errorhandler(500)
    def internal_error(_error):
        app.logger.exception("Unhandled FanCart request error.")
        return api_error("An internal server error occurred.", 500)

    @app.errorhandler(ModelUnavailableError)
    def model_unavailable(error):
        return api_error(str(error), 503)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.post("/api/auth/signup")
    def auth_signup():
        body = json_body()
        if body is None:
            return api_error("A JSON request body is required.", 400)
        username, password = body.get("username"), body.get("password")
        if not isinstance(username, str) or not USERNAME_PATTERN.fullmatch(username.strip()):
            return api_error(
                "Username must be 3-64 characters using letters, numbers, dots, dashes or underscores.",
                400,
            )
        if username.strip().casefold() == "demo_fan":
            return api_error("That username is reserved.", 409)
        if not isinstance(password, str) or not 6 <= len(password) <= 256:
            return api_error("Password must be between 6 and 256 characters.", 400)
        result = signup(username, password, app.config.get("DATABASE"))
        if not result.success:
            return api_error(result.message, 409 if "already in use" in result.message else 400)
        session.clear()
        session["username"] = result.username
        return jsonify({"user": {"username": result.username}, "message": result.message}), 201

    @app.post("/api/auth/login")
    def auth_login():
        body = json_body()
        if body is None:
            return api_error("A JSON request body is required.", 400)
        username, password = body.get("username"), body.get("password")
        remember = body.get("remember", False)
        if not isinstance(username, str) or not isinstance(password, str):
            return api_error("Username and password are required.", 400)
        if not isinstance(remember, bool):
            return api_error("Remember must be true or false.", 400)
        if len(username) > 64 or len(password) > 256:
            return api_error("Username or password is too long.", 400)
        result = login(username, password, remember, app.config.get("DATABASE"))
        if not result.success or result.username is None:
            return api_error(result.message, 401)
        session.clear()
        session["username"] = result.username
        response = jsonify({"user": {"username": result.username}, "message": result.message})
        previous_token = request.cookies.get(REMEMBER_COOKIE)
        if previous_token:
            revoke_token(previous_token, app.config.get("DATABASE"))
        if result.token:
            response.set_cookie(
                REMEMBER_COOKIE,
                result.token,
                max_age=TOKEN_LIFETIME_SECONDS,
                httponly=True,
                secure=bool(app.config["REMEMBER_COOKIE_SECURE"]),
                samesite="Lax",
                path="/",
            )
        else:
            response.delete_cookie(REMEMBER_COOKIE, path="/", samesite="Lax")
        return response

    @app.post("/api/auth/demo")
    def auth_demo():
        username = ensure_demo_user(app.config.get("DATABASE"))
        session.clear()
        session["username"] = username
        return jsonify({"user": {"username": username}, "message": "Demo session started."})

    @app.post("/api/auth/logout")
    def auth_logout():
        token = request.cookies.get(REMEMBER_COOKIE)
        if token:
            revoke_token(token, app.config.get("DATABASE"))
        session.clear()
        response = jsonify({"message": "Logged out."})
        response.delete_cookie(REMEMBER_COOKIE, path="/", samesite="Lax")
        response.delete_cookie(app.config.get("SESSION_COOKIE_NAME", "session"), path="/", samesite="Lax")
        return response

    @app.get("/api/auth/me")
    def auth_me():
        username = authenticated_username()
        if username is None:
            response, status = api_error("Authentication required.", 401)
            if request.cookies.get(REMEMBER_COOKIE):
                response.delete_cookie(REMEMBER_COOKIE, path="/", samesite="Lax")
            return response, status
        session["username"] = username
        return jsonify({"user": {"username": username}})

    @app.get("/api/data/status")
    def data_status():
        username = authenticated_username()
        if username is None or is_demo_user(username):
            return jsonify(
                {
                    "source": "Source: Synthetic (sample data)",
                    "record_count": 0,
                    "demo_user": is_demo_user(username),
                    "message": demo_data_error["error"] if is_demo_user(username) else None,
                }
            )
        count = user_data.record_count(
            current_data_user_id(), app.config.get("DATABASE")
        )
        return jsonify(
            {
                "source": "Source: Uploaded data" if count else "Source: Synthetic (sample data)",
                "record_count": count,
                "demo_user": False,
                "message": None,
            }
        )

    @app.post("/api/data/upload")
    def data_upload():
        if is_demo_user():
            return jsonify(demo_data_error), 403
        uploaded_file = request.files.get("file")
        if uploaded_file is None or not uploaded_file.filename:
            return api_error("Choose a CSV file to upload.", 400)
        try:
            frame = pd.read_csv(io.BytesIO(uploaded_file.read()))
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            return api_error("The uploaded file must be a readable CSV file.", 400)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        required = {"date", "team", "product", "units"}
        if not required.issubset(frame.columns):
            return api_error("CSV must contain date, team, product and units columns.", 400)
        if frame.empty:
            return api_error("The CSV must contain at least one sales record.", 400)
        if len(frame) > 100_000:
            return api_error("Upload no more than 100000 sales records at a time.", 413)
        try:
            records = [
                validated_sales_row(row)
                for row in frame.loc[:, ["date", "team", "product", "units"]].to_dict("records")
            ]
        except ValueError as error:
            return api_error(str(error), 400)
        user_data.replace_records(
            current_data_user_id(), records, app.config.get("DATABASE")
        )
        return jsonify(
            {
                "message": f"Uploaded {len(records)} sales records.",
                "record_count": user_data.record_count(
                    current_data_user_id(), app.config.get("DATABASE")
                ),
                "source": "Source: Uploaded data",
            }
        )

    @app.post("/api/data/add-row")
    def data_add_row():
        if is_demo_user():
            return jsonify(demo_data_error), 403
        body = json_body()
        if body is None:
            return api_error("A JSON request body is required.", 400)
        try:
            row = validated_sales_row(body)
        except ValueError as error:
            return api_error(str(error), 400)
        user_id = current_data_user_id()
        user_data.upsert_record(user_id, row, app.config.get("DATABASE"))
        count = user_data.record_count(user_id, app.config.get("DATABASE"))
        return jsonify(
            {
                "message": "Sales record saved.",
                "record_count": count,
                "source": "Source: Uploaded data",
            }
        ), 201

    @app.post("/api/data/reset")
    def data_reset():
        if is_demo_user():
            return jsonify(demo_data_error), 403
        user_data.reset(current_data_user_id(), app.config.get("DATABASE"))
        return jsonify(
            {
                "message": "Your data was reset to the sample data.",
                "record_count": 0,
                "source": "Source: Synthetic (sample data)",
            }
        )

    @app.get("/api/meta")
    def meta():
        state = current_runtime()
        fixtures = state.fixtures.copy()
        today = pd.Timestamp.today().normalize()
        fixtures = fixtures[
            (pd.to_datetime(fixtures["date"]) >= today)
            & (pd.to_datetime(fixtures["date"]) <= today + pd.Timedelta(30, unit="D"))
        ].sort_values("date")
        return jsonify(
            {
                "teams": list(ACTIVE_TEAMS),
                "products": list(PRODUCTS),
                "fixtures": [
                    {
                        "date": pd.Timestamp(row.date).date().isoformat(),
                        "team1": row.team1,
                        "team2": row.team2,
                    }
                    for row in fixtures.itertuples(index=False)
                ],
                "data_source": f"Match data: {state.source}",
                "sales_note": "Sales data is synthetic.",
            }
        )

    @app.get("/api/kpis")
    def kpis():
        state = current_runtime()
        today = pd.Timestamp.today().normalize()
        combinations = [(team, product) for team in ACTIVE_TEAMS for product in PRODUCTS]
        next_matches = {}
        for team in ACTIVE_TEAMS:
            relevant = state.fixtures[
                ((state.fixtures["team1"] == team) | (state.fixtures["team2"] == team))
                & (pd.to_datetime(state.fixtures["date"]) > today)
            ]
            if not relevant.empty:
                next_matches[team] = pd.to_datetime(relevant["date"]).min()
        forecasts = forecast_products_batch(
            state.sales,
            state.matches,
            state.fixtures,
            state.model,
            combinations,
            today + pd.Timedelta(1, unit="D"),
            days=14,
            scenario_match_dates=next_matches,
        )
        total_units = float(forecasts["median"].sum())
        risk_count = 0
        for team, product in combinations:
            forecast = forecasts[
                (forecasts["team"] == team) & (forecasts["product"] == product)
            ]
            recent = state.sales[
                (state.sales["team"] == team)
                & (state.sales["product"] == product)
                & (pd.to_datetime(state.sales["date"]) >= today - pd.Timedelta(28, unit="D"))
            ]
            lead_need = float(recent["units"].mean()) * 7 if not recent.empty else 0.0
            if recommend_stock(forecast, 7) < lead_need:
                risk_count += 1
        uplift = 0.0
        upcoming = state.fixtures[
            pd.to_datetime(state.fixtures["date"]) > today
        ].sort_values("date")
        if not upcoming.empty:
            fixture = upcoming.iloc[0]
            team = fixture["team1"] if fixture["team1"] in ACTIVE_TEAMS else fixture["team2"]
            match_day = pd.Timestamp(fixture["date"]).normalize()
            start = today + pd.Timedelta(1, unit="D")
            horizon = max(7, (match_day - today).days + 8)
            expected = forecast_product(
                state.sales, state.matches, state.fixtures, state.model,
                team, PRODUCTS[0], start, horizon, "expected", match_day, recursive=False,
            )
            winning = forecast_product(
                state.sales, state.matches, state.fixtures, state.model,
                team, PRODUCTS[0], start, horizon, "win", match_day, recursive=False,
            )
            post_match = (winning["date"] > match_day) & (
                winning["date"] <= match_day + pd.Timedelta(7, unit="D")
            )
            expected_total = float(expected.loc[post_match, "median"].sum())
            uplift = (
                float(winning.loc[post_match, "median"].sum()) - expected_total
            ) / max(expected_total, 1.0) * 100
        return jsonify(
            {
                "forecast_units_14_days": total_units,
                "products_at_stockout_risk": risk_count,
                "next_match_uplift_pct": uplift,
            }
        )

    @app.get("/api/forecast")
    def forecast():
        state = current_runtime()
        team, product = request.args.get("team"), request.args.get("product")
        if team not in ACTIVE_TEAMS:
            return api_error("Select a valid team.", 400)
        if product not in PRODUCTS:
            return api_error("Select a valid product.", 400)
        try:
            days = int(request.args.get("days", "14"))
        except ValueError:
            return api_error("Days must be an integer from 1 to 60.", 400)
        if not 1 <= days <= 60:
            return api_error("Days must be an integer from 1 to 60.", 400)
        today = pd.Timestamp.today().normalize()
        upcoming = state.fixtures[
            ((state.fixtures["team1"] == team) | (state.fixtures["team2"] == team))
            & (pd.to_datetime(state.fixtures["date"]) > today)
        ]
        match_date = pd.to_datetime(upcoming["date"]).min() if not upcoming.empty else None
        predicted = forecast_product(
            state.sales,
            state.matches,
            state.fixtures,
            state.model,
            team,
            product,
            today + pd.Timedelta(1, unit="D"),
            days,
            "expected",
            match_date,
        )
        history = state.sales[
            (state.sales["team"] == team)
            & (state.sales["product"] == product)
            & (pd.to_datetime(state.sales["date"]) >= today - pd.Timedelta(89, unit="D"))
            & (pd.to_datetime(state.sales["date"]) <= today)
        ].sort_values("date")
        return jsonify(
            {
                "team": team,
                "product": product,
                "history": [
                    {"date": pd.Timestamp(row.date).date().isoformat(), "units": int(row.units)}
                    for row in history.itertuples(index=False)
                ],
                "forecast": [
                    {
                        "date": pd.Timestamp(row.date).date().isoformat(),
                        "median": float(row.median),
                        "lower": float(row.lower),
                        "upper": float(row.upper),
                    }
                    for row in predicted.itertuples(index=False)
                ],
            }
        )

    @app.post("/api/whatif")
    def whatif():
        state = current_runtime()
        body = json_body()
        if body is None:
            return api_error("A JSON request body is required.", 400)
        team, product = body.get("team"), body.get("product")
        match_date_value = body.get("match_date")
        outcome = body.get("outcome")
        lead_time = body.get("lead_time")
        if team not in ACTIVE_TEAMS:
            return api_error("Select a valid team.", 400)
        if product not in PRODUCTS:
            return api_error("Select a valid product.", 400)
        if not isinstance(outcome, str) or outcome not in {"win", "loss"}:
            return api_error("Outcome must be win or loss.", 400)
        if isinstance(lead_time, bool) or not isinstance(lead_time, int) or not 1 <= lead_time <= 60:
            return api_error("Lead time must be an integer from 1 to 60 days.", 400)
        try:
            match_date = pd.Timestamp(date.fromisoformat(match_date_value)).normalize()
        except (TypeError, ValueError):
            return api_error("Match date must be a valid ISO date.", 400)
        today = pd.Timestamp.today().normalize()
        fixture_exists = (
            state.fixtures["date"].dt.normalize().eq(match_date)
            & ((state.fixtures["team1"] == team) | (state.fixtures["team2"] == team))
        ).any()
        if match_date <= today or match_date > today + pd.Timedelta(30, unit="D") or not fixture_exists:
            return api_error("Choose an upcoming fixture for the selected team.", 400)
        result = compare_scenarios(
            state.sales, state.matches, state.fixtures, state.model,
            team, product, match_date, lead_time,
        )
        match_day_mask = (result["win"]["date"] > match_date) & (
            result["win"]["date"] <= match_date + pd.Timedelta(7, unit="D")
        )
        win_demand = float(result["win"].loc[match_day_mask, "median"].sum())
        loss_demand = float(result["loss"].loc[match_day_mask, "median"].sum())
        baseline_demand = float(result["baseline"].loc[match_day_mask, "median"].sum())
        change_vs_baseline = (
            (win_demand if outcome == "win" else loss_demand) - baseline_demand
        ) / max(baseline_demand, 1.0) * 100
        recommended = recommend_stock(result[outcome], lead_time)
        day_name = match_date.strftime("%A")
        win_change_pct = (win_demand - loss_demand) / max(loss_demand, 1.0) * 100
        sentence = (
            f"If {team} wins {day_name}, {product} demand rises {win_change_pct:.1f}% "
            f"for {result['uplift_days']} days. Recommended stock: "
            f"{result['recommended_stock']} units instead of {result['baseline_stock']}."
        )
        return jsonify(
            {
                "outcome": outcome,
                "demand_change_pct": float(change_vs_baseline),
                "win_demand_7_days": win_demand,
                "loss_demand_7_days": loss_demand,
                "baseline_demand_7_days": baseline_demand,
                "days_until_uplift_below_5_pct": int(result["uplift_days"]),
                "recommended_stock": int(recommended),
                "win_recommended_stock": int(result["recommended_stock"]),
                "baseline_stock": int(result["baseline_stock"]),
                "sentence": sentence,
            }
        )

    @app.get("/api/report")
    def report():
        state = current_runtime()
        metrics_path = APP_ROOT / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            app.logger.exception("Unable to read metrics.json.")
            return api_error("Model metrics are unavailable.", 503)
        sample = state.model["validation_sample"]
        importance = permutation_importance(
            state.model["median"],
            sample.loc[:, FEATURE_COLUMNS],
            sample["units"],
            n_repeats=3,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )
        ranked = sorted(
            [
                {"feature": feature, "importance": float(value)}
                for feature, value in zip(FEATURE_COLUMNS, importance.importances_mean, strict=True)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )
        return jsonify(
            {
                "validation_cutoff": metrics.get("validation_cutoff"),
                "model": metrics["model"],
                "seasonal_naive": metrics["seasonal_naive"],
                "feature_importance": ranked,
                "note": "Sales data is synthetic.",
            }
        )

    @app.get("/api/insights")
    def insights():
        state = current_runtime()
        history = state.matches.copy()
        history["date"] = pd.to_datetime(history["date"]).dt.normalize()
        history = history[history["winner"].notna()]
        team_results = pd.concat(
            [
                history[["date", "team1", "winner"]].rename(columns={"team1": "team"}),
                history[["date", "team2", "winner"]].rename(columns={"team2": "team"}),
            ],
            ignore_index=True,
        )
        team_results["result"] = np.where(team_results["team"] == team_results["winner"], "win", "loss")
        post_match_records = [
            {
                "date": row.date + pd.Timedelta(offset, unit="D"),
                "team": row.team,
                "result": row.result,
                "match_date": row.date,
            }
            for row in team_results.itertuples(index=False)
            for offset in range(1, 8)
        ]
        post_match = pd.DataFrame(
            post_match_records,
            columns=["date", "team", "result", "match_date"],
        )
        if not post_match.empty:
            post_match = (
                post_match.sort_values(["date", "team", "match_date"])
                .drop_duplicates(["date", "team"], keep="last")
                .drop(columns="match_date")
            )
        else:
            post_match["date"] = pd.to_datetime(post_match["date"])
        sales = state.sales.copy()
        sales["date"] = pd.to_datetime(sales["date"]).dt.normalize()
        expanded = sales.merge(
            post_match,
            on=["date", "team"],
            how="left",
        )
        match_days = team_results[["date", "team"]].drop_duplicates(["date", "team"])
        match_day = sales.merge(match_days, on=["date", "team"], how="inner")
        non_match_day = sales.merge(
            match_days.assign(is_match_day=True),
            on=["date", "team"],
            how="left",
        )
        non_match_day = non_match_day[non_match_day["is_match_day"].isna()]
        win_values = expanded.loc[expanded["result"] == "win", "units"]
        loss_values = expanded.loc[expanded["result"] == "loss", "units"]
        return jsonify(
            {
                "average_demand_after_wins": float(win_values.mean()) if not win_values.empty else 0.0,
                "average_demand_after_losses": float(loss_values.mean()) if not loss_values.empty else 0.0,
                "average_match_day_demand": float(match_day["units"].mean()) if not match_day.empty else 0.0,
                "average_non_match_day_demand": float(non_match_day["units"].mean()) if not non_match_day.empty else 0.0,
                "note": "Insights are descriptive comparisons from synthetic sales data.",
            }
        )

    model_path = Path(app.config["FANCART_MODEL_DIR"]) / "fancart_model.joblib"
    if not model_path.is_file():
        training_status["started"] = True
        threading.Thread(
            target=train_runtime_in_background,
            name="fancart-model-training",
            daemon=True,
        ).start()
    return app


app = create_app()


if __name__ == "__main__":
    port_value = os.environ.get("PORT")
    app.run(
        host="0.0.0.0" if port_value is not None else "127.0.0.1",
        port=int(port_value) if port_value is not None else 5000,
        debug=False,
    )
