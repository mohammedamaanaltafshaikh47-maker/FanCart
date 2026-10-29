from __future__ import annotations

import numpy as np
import pandas as pd

from ml.predict import forecast_product, forecast_products_batch, recommend_stock


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value)


class FeatureModel:
    def __init__(self, feature: str):
        self.feature = feature

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.feature].to_numpy(dtype=float)


def test_recommended_stock_is_nonnegative_and_monotonic() -> None:
    forecast = pd.DataFrame(
        {
            "median": [10.0, 11.0, 12.0, 13.0],
            "upper": [14.0, 15.0, 16.0, 17.0],
        }
    )

    short_lead = recommend_stock(forecast, 2)
    long_lead = recommend_stock(forecast, 4)

    assert short_lead >= 0
    assert long_lead >= short_lead
    assert recommend_stock(forecast, 0) == 0


def test_forecast_runs_with_encoded_features() -> None:
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    sales = pd.DataFrame(
        {
            "date": dates,
            "team": "Chennai Super Kings",
            "product": "jersey",
            "units": np.arange(35) + 10,
        }
    )
    model = {
        "median": ConstantModel(20),
        "lower": ConstantModel(15),
        "upper": ConstantModel(25),
    }

    forecast = forecast_product(
        sales,
        pd.DataFrame(columns=["date", "team1", "team2", "winner"]),
        pd.DataFrame(columns=["date", "team1", "team2"]),
        model,
        "Chennai Super Kings",
        "jersey",
        pd.Timestamp("2025-02-05"),
        days=3,
    )

    assert len(forecast) == 3
    assert (forecast["lower"] <= forecast["median"]).all()
    assert (forecast["median"] <= forecast["upper"]).all()


def test_future_match_result_does_not_change_forecast() -> None:
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    sales = pd.DataFrame(
        {
            "date": dates,
            "team": "Chennai Super Kings",
            "product": "jersey",
            "units": np.arange(35) + 10,
        }
    )
    match = pd.DataFrame(
        [{
            "date": "2025-02-06",
            "team1": "Chennai Super Kings",
            "team2": "Mumbai Indians",
            "winner": "Chennai Super Kings",
        }]
    )
    changed = match.copy()
    changed.loc[0, "winner"] = "Mumbai Indians"
    model = {
        "median": FeatureModel("last_match_won"),
        "lower": FeatureModel("last_match_won"),
        "upper": FeatureModel("last_match_won"),
    }

    first = forecast_product(
        sales, match, pd.DataFrame(), model, "Chennai Super Kings", "jersey",
        pd.Timestamp("2025-02-05"), days=4
    )
    second = forecast_product(
        sales, changed, pd.DataFrame(), model, "Chennai Super Kings", "jersey",
        pd.Timestamp("2025-02-05"), days=4
    )

    pd.testing.assert_frame_equal(first, second)


def test_batch_forecast_matches_single_nonrecursive_forecast() -> None:
    dates = pd.date_range("2025-01-01", periods=35, freq="D")
    sales = pd.DataFrame(
        {
            "date": dates,
            "team": "Chennai Super Kings",
            "product": "jersey",
            "units": np.arange(35) + 10,
        }
    )
    empty_matches = pd.DataFrame(
        columns=["date", "team1", "team2", "winner"]
    )
    empty_fixtures = pd.DataFrame(columns=["date", "team1", "team2"])
    model = {
        "median": ConstantModel(20),
        "lower": ConstantModel(15),
        "upper": ConstantModel(25),
    }

    single = forecast_product(
        sales,
        empty_matches,
        empty_fixtures,
        model,
        "Chennai Super Kings",
        "jersey",
        pd.Timestamp("2025-02-05"),
        days=3,
        recursive=False,
    )
    batch = forecast_products_batch(
        sales,
        empty_matches,
        empty_fixtures,
        model,
        [("Chennai Super Kings", "jersey")],
        pd.Timestamp("2025-02-05"),
        days=3,
    )

    pd.testing.assert_frame_equal(
        single,
        batch.drop(columns=["team", "product"]).reset_index(drop=True),
        check_dtype=False,
    )
