from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

RANDOM_SEED = 42
TEAMS = (
    "Chennai Super Kings",
    "Mumbai Indians",
    "Kolkata Knight Riders",
    "Royal Challengers Bengaluru",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
)
ACTIVE_TEAMS = TEAMS[:6]
PRODUCTS = ("jersey", "cap", "flag")
VENUES = (
    "Chennai",
    "Mumbai",
    "Kolkata",
    "Bengaluru",
    "Jaipur",
    "Hyderabad",
    "Delhi",
    "Mohali",
)

EFFECT_CONFIG = {
    "win_lift": 0.35,
    "win_decay_days": 6,
    "loss_dip": -0.10,
    "match_day_bump": 0.16,
    "weekend_effect": 0.12,
    "star_player_form_bump": 0.08,
}


def _season_matches(
    year: int, teams: tuple[str, ...], rng: np.random.Generator
) -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = []
    start = date(year, 3, 25)
    ratings = {team: float(rng.normal(0.5, 0.12)) for team in teams}
    for day_index in range(52):
        first = teams[day_index % len(teams)]
        second = teams[(day_index * 3 + day_index // len(teams) + 1) % len(teams)]
        if first == second:
            second = teams[(teams.index(second) + 1) % len(teams)]
        probability = float(np.clip(0.5 + ratings[first] - ratings[second], 0.25, 0.75))
        winner = first if rng.random() < probability else second
        loser = second if winner == first else first
        margin_type = "runs" if rng.random() < 0.52 else "wickets"
        schedule.append(
            {
                "date": start + timedelta(days=day_index),
                "team1": first,
                "team2": second,
                "venue": VENUES[TEAMS.index(first)],
                "winner": winner,
                "margin": f"{int(rng.integers(4, 68))} {margin_type}",
                "season": str(year),
            }
        )
        ratings[winner] = min(0.8, ratings[winner] + 0.025)
        ratings[loser] = max(0.2, ratings[loser] - 0.018)
    return schedule


def generate_matches() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    current_year = date.today().year
    for year in range(current_year - 2, current_year + 1):
        rows.extend(_season_matches(year, TEAMS, rng))
    result = pd.DataFrame(rows)
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values("date").reset_index(drop=True)


def generate_fixtures(
    teams: tuple[str, ...] = ACTIVE_TEAMS,
    start: date | None = None,
    days: int = 30,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    first_day = start or date.today()
    fixtures = []
    for offset in range(1, days + 1):
        if offset % 2:
            first = teams[(offset // 2) % len(teams)]
            second = teams[(offset // 2 + 3) % len(teams)]
            if first == second:
                second = teams[(teams.index(second) + 1) % len(teams)]
            fixtures.append(
                {
                    "date": pd.Timestamp(first_day + timedelta(days=offset)),
                    "team1": first,
                    "team2": second,
                    "venue": VENUES[TEAMS.index(first)],
                    "winner": None,
                    "margin": None,
                    "season": "simulated",
                    "simulated": True,
                    "fixture_id": f"sim-{offset}",
                    "form_noise": float(rng.normal(0, 0.01)),
                }
            )
    return pd.DataFrame(fixtures)


def generate_sales(
    matches: pd.DataFrame | None = None,
    start: date | None = None,
    end: date | None = None,
    teams: tuple[str, ...] = ACTIVE_TEAMS,
    random_seed: int | None = None,
    team_level_scale: tuple[float, float] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        RANDOM_SEED + 2 if random_seed is None else random_seed
    )
    last_day = end or date.today()
    first_day = start or last_day - timedelta(days=730)
    dates = pd.date_range(first_day, last_day, freq="D")
    base_demand = {"jersey": 23.0, "cap": 13.0, "flag": 9.0}
    price = {"jersey": 42.0, "cap": 18.0, "flag": 12.0}
    team_strength = {
        team: float(
            rng.uniform(0.9, 1.15)
            * (rng.uniform(*team_level_scale) if team_level_scale else 1.0)
        )
        for team in teams
    }
    history = matches if matches is not None else generate_matches()
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    rows: list[dict[str, object]] = []
    for team in teams:
        team_matches = history[
            (history["team1"] == team) | (history["team2"] == team)
        ].sort_values("date")
        for product in PRODUCTS:
            previous_result: int | None = None
            previous_result_date: pd.Timestamp | None = None
            match_dates = team_matches["date"].tolist()
            match_winners = team_matches["winner"].tolist()
            match_index = 0
            for day in dates:
                normalized = pd.Timestamp(day).normalize()
                match_today = False
                while match_index < len(match_dates) and match_dates[match_index] <= normalized:
                    match_today = match_dates[match_index] == normalized
                    previous_result = int(match_winners[match_index] == team)
                    previous_result_date = match_dates[match_index]
                    match_index += 1
                seasonal = 1 + 0.18 * np.sin(
                    2 * np.pi * (day.dayofyear - 95) / 365.25
                )
                expected = base_demand[product] * team_strength[team] * seasonal
                if day.weekday() >= 5:
                    expected *= 1 + EFFECT_CONFIG["weekend_effect"]
                if match_today:
                    expected *= 1 + EFFECT_CONFIG["match_day_bump"]
                star_form = float(
                    np.clip(0.50 + 0.20 * np.sin(2 * np.pi * day.dayofyear / 29), 0, 1)
                )
                expected *= 1 + EFFECT_CONFIG["star_player_form_bump"] * star_form
                if previous_result_date is not None and not match_today:
                    elapsed = (normalized - previous_result_date).days
                    if previous_result == 1 and elapsed <= EFFECT_CONFIG["win_decay_days"]:
                        decay = 1 - elapsed / (EFFECT_CONFIG["win_decay_days"] + 1)
                        expected *= 1 + EFFECT_CONFIG["win_lift"] * decay
                    elif previous_result == 0 and elapsed == 1:
                        expected *= 1 + EFFECT_CONFIG["loss_dip"]
                units = max(0, int(round(rng.normal(expected, max(2.0, expected * 0.16)))))
                rows.append(
                    {
                        "date": normalized,
                        "team": team,
                        "product": product,
                        "units": units,
                        "price": price[product],
                    }
                )
    return pd.DataFrame(rows).sort_values(["team", "product", "date"]).reset_index(drop=True)
