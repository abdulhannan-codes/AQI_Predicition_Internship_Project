"""Model loading and multi-horizon AQI prediction."""
import json
import pathlib

import joblib
import numpy as np
import pandas as pd

import models_def  # noqa: F401 — joblib unpickling
from config import HORIZONS
from services.aqi import aqi_category

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_registry() -> dict:
    path = MODELS_DIR / "registry.json"
    if not path.exists():
        raise FileNotFoundError("No registry.json — run python train.py first")
    return json.loads(path.read_text())


def horizon_feature_cols(registry: dict, horizon: int) -> list[str]:
    return registry["by_horizon"][str(horizon)]["feature_cols"]


def feature_vector(row: pd.Series, feature_cols: list[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in row.index]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return row[feature_cols].astype(np.float64).values.reshape(1, -1)


def load_horizon_model(horizon: int):
    return joblib.load(MODELS_DIR / f"best_model_h{horizon}.pkl")


def load_named_model(model_name: str, horizon: int):
    path = MODELS_DIR / f"{model_name}_h{horizon}.pkl"
    if path.exists():
        return joblib.load(path)
    return load_horizon_model(horizon)


def predict_horizon(latest: pd.Series, registry: dict, horizon: int) -> dict:
    cols = horizon_feature_cols(registry, horizon)
    X = feature_vector(latest, cols)
    model = load_horizon_model(horizon)
    pred = max(0.0, float(model.predict(X)[0]))
    h_info = registry["by_horizon"][str(horizon)]
    rmse = h_info["best_metrics"]["rmse"]
    return {
        "horizon_hours": horizon,
        "forecast_aqi": round(pred),
        "lower": max(0, round(pred - rmse)),
        "upper": round(pred + rmse),
        "best_model": h_info["best_model"],
        "rmse": rmse,
    }


def current_aqi(latest: pd.Series) -> dict:
    aqi = int(round(latest["aqi"]))
    label, color = aqi_category(aqi)
    return {"aqi": aqi, "category": label, "color": color}


def full_forecast(latest: pd.Series, registry: dict | None = None) -> dict:
    registry = registry or load_registry()
    current = current_aqi(latest)
    points = [{"hour_ahead": 0, **current, "forecast_aqi": current["aqi"],
                 "lower": current["aqi"], "upper": current["aqi"]}]
    for h in HORIZONS:
        p = predict_horizon(latest, registry, h)
        points.append({"hour_ahead": h, "forecast_aqi": p["forecast_aqi"],
                       "lower": p["lower"], "upper": p["upper"],
                       "best_model": p["best_model"]})
    peak = max(pt["forecast_aqi"] for pt in points)
    return {
        "current": current,
        "forecast": points,
        "peak_aqi": peak,
        "registry_trained_at": registry.get("trained_at"),
    }


def model_metrics(registry: dict | None = None) -> dict:
    registry = registry or load_registry()
    return registry.get("by_horizon", registry)
