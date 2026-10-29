from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable

import numpy as np
import pandas as pd

from data.synthetic import PRODUCTS

FEATURE_COLUMNS = (
    "days_to_next_match",
    "days_since_last_match",
    "last_match_won",
    "team_form_5",
    "win_streak",
    "is_weekend",
    "day_of_week",
    "month",
    "product",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_28",
)


def _match_records(matches: pd.DataFrame, team: str) -> list[tuple[pd.Timestamp, bool]]:
    if matches.empty:
        return []
    relevant = matches[
        (matches["team1"] == team) | (matches["team2"] == team)
    ].copy()
    relevant["date"] = pd.to_datetime(relevant["date"]).dt.normalize()
    relevant = relevant.sort_values("date")
    return [
        (row.date, row.winner == team)
        for row in relevant.itertuples(index=False)
        if isinstance(row.winner, str)
    ]


def _fixture_dates(
    fixtures: pd.DataFrame, matches: pd.DataFrame, team: str
) -> list[pd.Timestamp]:
    sources = []
    for frame in (fixtures, matches):
        if frame.empty:
            continue
        relevant = frame[(frame["team1"] == team) | (frame["team2"] == team)]
        if not relevant.empty:
            sources.extend(pd.to_datetime(relevant["date"]).dt.normalize().tolist())
    return sorted(set(sources))


def _recent_features(
    history: list[tuple[pd.Timestamp, bool]],
    dates: list[pd.Timestamp],
    target: pd.Timestamp,
) -> tuple[int, int, float, int]:
    index = bisect_left(dates, target)
    prior = history[:index]
    if not prior:
        return 30, 0, 0.5, 0
    last_date, last_won = prior[-1]
    recent = [result for _, result in prior[-5:]]
    streak = 0
    for _, won in reversed(prior):
        if not won:
            break
        streak += 1
    return (target - last_date).days, int(last_won), float(np.mean(recent)), streak


def build_features(
    sales: pd.DataFrame,
    matches: pd.DataFrame,
    fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if sales.empty:
        return sales.copy()
    required = {"date", "team", "product"}
    missing = required.difference(sales.columns)
    if missing:
        raise ValueError(f"Sales data is missing columns: {', '.join(sorted(missing))}")
    fixture_frame = fixtures if fixtures is not None else pd.DataFrame()
    frame = sales.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values(["team", "product", "date"]).reset_index(drop=True)
    if "units" not in frame:
        frame["units"] = np.nan

    grouped = frame.groupby(["team", "product"], sort=False)["units"]
    frame["lag_7"] = grouped.shift(7)
    prior_units = grouped.shift(1)
    frame["rolling_mean_7"] = prior_units.groupby(
        [frame["team"], frame["product"]], sort=False
    ).transform(lambda values: values.rolling(7, min_periods=1).mean())
    frame["rolling_mean_28"] = prior_units.groupby(
        [frame["team"], frame["product"]], sort=False
    ).transform(lambda values: values.rolling(28, min_periods=1).mean())

    product_codes = {product: index for index, product in enumerate(PRODUCTS)}
    frame["product"] = frame["product"].map(product_codes).fillna(-1).astype(int)
    frame["is_weekend"] = (frame["date"].dt.weekday >= 5).astype(int)
    frame["day_of_week"] = frame["date"].dt.weekday.astype(int)
    frame["month"] = frame["date"].dt.month.astype(int)
    frame["days_to_next_match"] = 30
    frame["days_since_last_match"] = 30
    frame["last_match_won"] = 0
    frame["team_form_5"] = 0.5
    frame["win_streak"] = 0

    for team in frame["team"].unique():
        team_mask = frame["team"] == team
        history = _match_records(matches, team)
        fixture_dates = _fixture_dates(fixture_frame, matches, team)
        target_dates = frame.loc[team_mask, "date"]
        match_dates = [item[0] for item in history]
        previous = [
            _recent_features(history, match_dates, target)
            for target in target_dates
        ]
        frame.loc[team_mask, "days_since_last_match"] = [item[0] for item in previous]
        frame.loc[team_mask, "last_match_won"] = [item[1] for item in previous]
        frame.loc[team_mask, "team_form_5"] = [item[2] for item in previous]
        frame.loc[team_mask, "win_streak"] = [item[3] for item in previous]
        if fixture_dates:
            next_days = []
            for target in target_dates:
                index = bisect_left(fixture_dates, target)
                next_days.append(
                    max(0, (fixture_dates[index] - target).days)
                    if index < len(fixture_dates)
                    else 30
                )
            frame.loc[team_mask, "days_to_next_match"] = next_days

    global_mean = float(frame["units"].mean()) if frame["units"].notna().any() else 0.0
    for column in ("lag_7", "rolling_mean_7", "rolling_mean_28"):
        frame[column] = frame[column].fillna(frame["rolling_mean_28"]).fillna(global_mean)
    frame[list(FEATURE_COLUMNS)] = frame[list(FEATURE_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    return frame


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Feature data is missing columns: {', '.join(sorted(missing))}")
    return frame.loc[:, FEATURE_COLUMNS].astype(float)
