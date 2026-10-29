from __future__ import annotations

import pandas as pd

from ml.features import build_features


def test_future_match_result_never_changes_prior_features() -> None:
    sales = pd.DataFrame(
        [
            {"date": "2025-01-01", "team": "Chennai Super Kings", "product": "jersey", "units": 12},
            {"date": "2025-01-02", "team": "Chennai Super Kings", "product": "jersey", "units": 14},
            {"date": "2025-01-03", "team": "Chennai Super Kings", "product": "jersey", "units": 13},
        ]
    )
    early = pd.DataFrame(
        [{"date": "2025-01-04", "team1": "Chennai Super Kings",
          "team2": "Mumbai Indians", "winner": "Mumbai Indians"}]
    )
    changed = early.copy()
    changed.loc[0, "winner"] = "Chennai Super Kings"

    original = build_features(sales, early).iloc[-1]
    altered = build_features(sales, changed).iloc[-1]

    assert original["last_match_won"] == altered["last_match_won"]
    assert original["team_form_5"] == altered["team_form_5"]
    assert original["win_streak"] == altered["win_streak"]


def test_prior_match_result_is_used_for_later_feature_rows() -> None:
    sales = pd.DataFrame(
        [{"date": "2025-01-03", "team": "Chennai Super Kings",
          "product": "cap", "units": 8}]
    )
    matches = pd.DataFrame(
        [{"date": "2025-01-01", "team1": "Chennai Super Kings",
          "team2": "Mumbai Indians", "winner": "Chennai Super Kings"}]
    )

    row = build_features(sales, matches).iloc[0]

    assert row["last_match_won"] == 1
    assert row["days_since_last_match"] == 2
