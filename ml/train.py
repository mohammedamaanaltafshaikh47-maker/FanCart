from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.features import FEATURE_COLUMNS, feature_matrix

RANDOM_SEED = 42
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE": float(
            np.mean(np.abs(actual - predicted) / np.maximum(np.abs(actual), 1.0))
            * 100
        ),
    }


def train_models(features: pd.DataFrame, model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    if "units" not in features or "date" not in features:
        raise ValueError("Training data must include units and date.")
    valid = features.dropna(subset=["units"]).copy()
    valid["date"] = pd.to_datetime(valid["date"])
    cutoff = valid["date"].max().normalize() - pd.Timedelta(days=59)
    train_rows = valid[valid["date"] < cutoff]
    test_rows = valid[valid["date"] >= cutoff]
    if train_rows.empty or test_rows.empty:
        raise ValueError("A time-based split requires training and validation rows.")

    train_x = feature_matrix(train_rows)
    test_x = feature_matrix(test_rows)
    train_y = train_rows["units"].astype(float).to_numpy()
    test_y = test_rows["units"].astype(float).to_numpy()
    options = {
        "max_iter": 90,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "l2_regularization": 0.5,
        "early_stopping": False,
        "random_state": RANDOM_SEED,
    }
    median_model = HistGradientBoostingRegressor(loss="absolute_error", **options)
    lower_model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.1, **options
    )
    upper_model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.9, **options
    )
    median_model.fit(train_x, train_y)
    lower_model.fit(train_x, train_y)
    upper_model.fit(train_x, train_y)

    median_prediction = median_model.predict(test_x)
    lower_prediction = lower_model.predict(test_x)
    upper_prediction = upper_model.predict(test_x)
    baseline_prediction = test_rows["lag_7"].astype(float).to_numpy()
    model_metrics = _metrics(test_y, median_prediction)
    baseline_metrics = _metrics(test_y, baseline_prediction)
    bundle: dict[str, Any] = {
        "model_version": 2,
        "median": median_model,
        "lower": lower_model,
        "upper": upper_model,
        "feature_names": list(FEATURE_COLUMNS),
        "metrics": {"model": model_metrics, "seasonal_naive": baseline_metrics},
        "validation_cutoff": cutoff.date().isoformat(),
        "validation_sample": test_rows.loc[:, [*FEATURE_COLUMNS, "units"]]
        .sample(n=min(300, len(test_rows)), random_state=RANDOM_SEED)
        .reset_index(drop=True),
    }
    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination / "fancart_model.joblib")
    with (destination.parent / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "validation_cutoff": bundle["validation_cutoff"],
                "model": model_metrics,
                "seasonal_naive": baseline_metrics,
            },
            file,
            indent=2,
        )
    return bundle


def load_or_train(
    features: pd.DataFrame, model_dir: str | Path = DEFAULT_MODEL_DIR
) -> dict[str, Any]:
    model_path = Path(model_dir) / "fancart_model.joblib"
    if model_path.exists():
        bundle = joblib.load(model_path)
        if bundle.get("model_version") == 2:
            return bundle
    return train_models(features, model_dir)
