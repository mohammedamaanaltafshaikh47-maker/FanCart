from __future__ import annotations

import math
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, build_features


def recommend_stock(forecast: pd.DataFrame, lead_time_days: int) -> int:
    if lead_time_days < 0:
        raise ValueError("Lead time must be zero or more days.")
    window = forecast.head(lead_time_days)
    if window.empty:
        return 0
    median = window["median"].clip(lower=0)
    safety = (window["upper"] - window["median"]).clip(lower=0)
    return max(0, int(math.ceil(float((median + safety).sum()))))


def _scenario_features(
    row: pd.Series,
    target_date: pd.Timestamp,
    scenario_result: str | None,
    match_date: pd.Timestamp | None,
) -> pd.Series:
    if (
        scenario_result not in {"win", "loss", "expected"}
        or match_date is None
        or target_date <= match_date
    ):
        return row
    if scenario_result == "expected":
        result_value = float(np.clip(row["team_form_5"], 0.15, 0.85))
    else:
        result_value = float(scenario_result == "win")
    row["days_since_last_match"] = (target_date - match_date).days
    row["last_match_won"] = result_value
    current_form = float(row["team_form_5"])
    row["team_form_5"] = float(
        np.clip(current_form + (result_value - current_form) / 5, 0, 1)
    )
    if scenario_result == "expected":
        row["win_streak"] = float(row["win_streak"]) + result_value
    else:
        row["win_streak"] = int(row["win_streak"]) + 1 if result_value else 0
    return row


def _model_row(
    features: pd.Series,
    target_date: pd.Timestamp,
    history_units: dict[pd.Timestamp, float],
    fallback: float,
    disable_match_effects: bool,
    scenario_result: str | None,
    match_date: pd.Timestamp | None,
) -> dict[str, float]:
    normalized_date = pd.Timestamp(target_date).normalize()
    previous_dates = [
        normalized_date - pd.to_timedelta(offset, unit="D")
        for offset in range(1, 29)
    ]
    previous_week = [
        history_units[day] for day in previous_dates[:7] if day in history_units
    ]
    previous_month = [
        history_units[day] for day in previous_dates if day in history_units
    ]
    features["lag_7"] = history_units.get(
        previous_dates[6], float(np.mean(previous_week)) if previous_week else fallback
    )
    features["rolling_mean_7"] = (
        float(np.mean(previous_week)) if previous_week else fallback
    )
    features["rolling_mean_28"] = (
        float(np.mean(previous_month)) if previous_month else fallback
    )
    if disable_match_effects:
        features["days_to_next_match"] = 30
        features["last_match_won"] = 0
        features["days_since_last_match"] = 30
        features["win_streak"] = 0
        features["team_form_5"] = 0.5
    else:
        features = _scenario_features(
            features, target_date, scenario_result, match_date
        )
    return {column: float(features[column]) for column in FEATURE_COLUMNS}


def forecast_product(
    sales: pd.DataFrame,
    matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    model: dict[str, Any],
    team: str,
    product: str,
    start: date | pd.Timestamp,
    days: int = 14,
    scenario_result: str | None = None,
    scenario_match_date: date | pd.Timestamp | None = None,
    disable_match_effects: bool = False,
    recursive: bool = True,
) -> pd.DataFrame:
    if days < 1:
        raise ValueError("Forecast horizon must be at least one day.")
    start_date = pd.Timestamp(start).normalize()
    match_date = (
        pd.Timestamp(scenario_match_date).normalize()
        if scenario_match_date is not None
        else None
    )
    historical = sales[
        (sales["team"] == team)
        & (sales["product"] == product)
        & (pd.to_datetime(sales["date"]) < start_date)
    ].copy()
    if historical.empty:
        raise ValueError(f"No historical sales are available for {team} {product}.")

    known_matches = matches.copy()
    if not known_matches.empty:
        future_match = (
            pd.to_datetime(known_matches["date"]).dt.normalize() >= start_date
        )
        known_matches.loc[future_match, "winner"] = None
    target_dates = pd.date_range(start_date, periods=days, freq="D")
    targets = pd.DataFrame(
        {
            "date": target_dates,
            "team": team,
            "product": product,
            "units": np.nan,
        }
    )
    feature_rows = build_features(targets, known_matches, fixtures)
    history_units = {
        pd.Timestamp(day).normalize(): float(units)
        for day, units in zip(historical["date"], historical["units"], strict=True)
    }
    fallback = float(historical["units"].mean())
    if not recursive:
        rows = [
            _model_row(
                feature.copy(),
                target_date,
                history_units,
                fallback,
                disable_match_effects,
                scenario_result,
                match_date,
            )
            for target_date, (_, feature) in zip(
                target_dates, feature_rows.iterrows(), strict=True
            )
        ]
        matrix = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        medians = np.maximum(0.0, model["median"].predict(matrix))
        lowers = model["lower"].predict(matrix)
        uppers = model["upper"].predict(matrix)
        return pd.DataFrame(
            {
                "date": target_dates,
                "median": medians,
                "lower": np.minimum(medians, np.maximum(0.0, lowers)),
                "upper": np.maximum(medians, uppers),
            }
        )

    output: list[dict[str, object]] = []
    model_rows: list[dict[str, float]] = []
    for (target_date, features) in zip(
        target_dates, (row.copy() for _, row in feature_rows.iterrows()), strict=True
    ):
        normalized_date = pd.Timestamp(target_date).normalize()
        matrix_row = _model_row(
            features,
            target_date,
            history_units,
            fallback,
            disable_match_effects,
            scenario_result,
            match_date,
        )
        matrix = pd.DataFrame([matrix_row], columns=FEATURE_COLUMNS)
        median = max(0.0, float(model["median"].predict(matrix)[0]))
        model_rows.append(matrix_row)
        output.append(
            {
                "date": target_date,
                "median": median,
            }
        )
        history_units[normalized_date] = median

    prediction_rows = pd.DataFrame(model_rows, columns=FEATURE_COLUMNS)
    lower_predictions = model["lower"].predict(prediction_rows)
    upper_predictions = model["upper"].predict(prediction_rows)
    for row, lower, upper in zip(
        output, lower_predictions, upper_predictions, strict=True
    ):
        row["lower"] = min(row["median"], max(0.0, float(lower)))
        row["upper"] = max(row["median"], float(upper))
    return pd.DataFrame(output)


def forecast_products_batch(
    sales: pd.DataFrame,
    matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    model: dict[str, Any],
    products: Sequence[tuple[str, str]],
    start: date | pd.Timestamp,
    days: int,
    scenario_match_dates: Mapping[str, date | pd.Timestamp] | None = None,
    scenario_result: str | None = "expected",
) -> pd.DataFrame:
    if days < 1:
        raise ValueError("Forecast horizon must be at least one day.")
    combinations = list(dict.fromkeys(products))
    if not combinations:
        return pd.DataFrame(columns=["team", "product", "date", "median", "lower", "upper"])

    start_date = pd.Timestamp(start).normalize()
    target_dates = pd.date_range(start_date, periods=days, freq="D")
    history_by_id: dict[int, dict[pd.Timestamp, float]] = {}
    fallback_by_id: dict[int, float] = {}
    target_records: list[dict[str, object]] = []
    for series_id, (team, product) in enumerate(combinations):
        history = sales[
            (sales["team"] == team)
            & (sales["product"] == product)
            & (pd.to_datetime(sales["date"]) < start_date)
        ].copy()
        if history.empty:
            raise ValueError(f"No historical sales are available for {team} {product}.")
        history_by_id[series_id] = {
            pd.Timestamp(day).normalize(): float(units)
            for day, units in zip(history["date"], history["units"], strict=True)
        }
        fallback_by_id[series_id] = float(history["units"].mean())
        target_records.extend(
            {
                "date": target_date,
                "team": team,
                "product": product,
                "units": np.nan,
                "series_id": series_id,
            }
            for target_date in target_dates
        )

    targets = pd.DataFrame(target_records)
    known_matches = matches.copy()
    if not known_matches.empty:
        future_match = (
            pd.to_datetime(known_matches["date"]).dt.normalize() >= start_date
        )
        known_matches.loc[future_match, "winner"] = None
    feature_rows = build_features(targets, known_matches, fixtures)
    match_dates = {
        team: pd.Timestamp(match_date).normalize()
        for team, match_date in (scenario_match_dates or {}).items()
    }
    matrix_rows: list[dict[str, float]] = []
    output_rows: list[dict[str, object]] = []
    for _, feature in feature_rows.iterrows():
        series_id = int(feature["series_id"])
        team, product = combinations[series_id]
        target_date = pd.Timestamp(feature["date"]).normalize()
        matrix_rows.append(
            _model_row(
                feature.copy(),
                target_date,
                history_by_id[series_id],
                fallback_by_id[series_id],
                False,
                scenario_result,
                match_dates.get(team),
            )
        )
        output_rows.append(
            {"team": team, "product": product, "date": target_date}
        )

    matrix = pd.DataFrame(matrix_rows, columns=FEATURE_COLUMNS)
    medians = np.maximum(0.0, model["median"].predict(matrix))
    lowers = model["lower"].predict(matrix)
    uppers = model["upper"].predict(matrix)
    for row, median, lower, upper in zip(
        output_rows, medians, lowers, uppers, strict=True
    ):
        row["median"] = float(median)
        row["lower"] = min(float(median), max(0.0, float(lower)))
        row["upper"] = max(float(median), float(upper))
    return pd.DataFrame(output_rows)


def compare_scenarios(
    sales: pd.DataFrame,
    matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    model: dict[str, Any],
    team: str,
    product: str,
    match_date: date | pd.Timestamp,
    lead_time_days: int,
) -> dict[str, Any]:
    start = pd.Timestamp.today().normalize()
    days_until_match = max(0, (pd.Timestamp(match_date).normalize() - start).days)
    horizon = max(14, lead_time_days, days_until_match + 8)
    win = forecast_product(
        sales, matches, fixtures, model, team, product, start, horizon, "win", match_date
    )
    loss = forecast_product(
        sales, matches, fixtures, model, team, product, start, horizon, "loss", match_date
    )
    baseline = forecast_product(
        sales,
        matches,
        fixtures,
        model,
        team,
        product,
        start,
        horizon,
        disable_match_effects=True,
    )
    match_day = pd.Timestamp(match_date).normalize()
    post_match = (win["date"] > match_day) & (
        win["date"] <= match_day + pd.Timedelta(7, unit="D")
    )
    win_week = float(win.loc[post_match, "median"].sum())
    loss_week = float(loss.loc[post_match, "median"].sum())
    change_pct = (win_week - loss_week) / max(loss_week, 1.0) * 100
    duration = 0
    for day, winning, losing in zip(
        win["date"], win["median"], loss["median"], strict=True
    ):
        if day <= match_day:
            continue
        uplift = (winning - losing) / max(losing, 1.0) * 100
        if uplift >= 5:
            duration += 1
        else:
            break
    return {
        "win": win,
        "loss": loss,
        "baseline": baseline,
        "change_pct": float(change_pct),
        "uplift_days": duration,
        "recommended_stock": recommend_stock(win, lead_time_days),
        "baseline_stock": recommend_stock(baseline, lead_time_days),
    }
