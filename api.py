"""
FastAPI backend — serves AQI predictions (#7, #29).

Run: uvicorn api:app --reload --port 8000
"""
from env_loader import load_env

load_env()

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.features import get_recent_features, latest_timestamp
from services.predict import full_forecast, load_registry, model_metrics
import feature_pipeline as fp

app = FastAPI(title="Pearls AQI Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "pearls-aqi-predictor"}


@app.get("/aqi/current")
def aqi_current():
    try:
        feats = get_recent_features()
        latest = feats.iloc[-1]
        result = full_forecast(latest)["current"]
        result["last_updated"] = str(latest_timestamp(feats))
        result["data_sources"] = dict(fp.LAST_DATA_SOURCES)
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/forecast")
def forecast():
    try:
        feats = get_recent_features()
        latest = feats.iloc[-1]
        out = full_forecast(latest)
        out["last_updated"] = str(latest_timestamp(feats))
        out["data_sources"] = dict(fp.LAST_DATA_SOURCES)
        out["feature_store"] = os.getenv("FEATURE_STORE", "local")
        return out
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/models/metrics")
def metrics():
    try:
        registry = load_registry()
        return {
            "trained_at": registry.get("trained_at"),
            "best_model": registry.get("best_model"),
            "by_horizon": model_metrics(registry),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/features/recent")
def features_recent(hours: int = 72):
    try:
        days = max(3, (hours // 24) + 3)
        feats = get_recent_features(days_back=days)
        tail = feats.tail(min(hours, len(feats)))
        return {
            "rows": len(tail),
            "last_updated": str(latest_timestamp(tail)),
            "data_sources": dict(fp.LAST_DATA_SOURCES),
            "data": tail.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
