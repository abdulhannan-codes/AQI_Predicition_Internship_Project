"""
app.py — Streamlit dashboard (UI layer).
Uses FastAPI when API_URL is set; otherwise calls services/ directly.

Run:  streamlit run app.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.pipeline import Pipeline

import feature_pipeline as fp
import models_def  # noqa: F401 — joblib unpickling
from config import HORIZONS
from services.aqi import aqi_category
from services.features import get_recent_features, latest_timestamp
from services.predict import (
    feature_vector,
    full_forecast,
    horizon_feature_cols,
    load_named_model,
    load_registry,
)

API_URL = os.getenv("API_URL", "").rstrip("/")

st.set_page_config(page_title="Pearls AQI Predictor — Lahore", layout="wide")


def fetch_from_api(path: str) -> dict:
    r = requests.get(f"{API_URL}{path}", timeout=120)
    r.raise_for_status()
    return r.json()


def render_shap(model, X_bg, X_pred, feature_names):
    import shap

    if hasattr(model, "aqi_col_idx"):
        st.info("Persistence model: forecast equals current AQI (single-feature baseline).")
        idx = model.aqi_col_idx
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh([feature_names[idx]], [X_pred[0, idx]])
        ax.set_xlabel("Contribution")
        st.pyplot(fig)
        plt.close(fig)
        return

    if isinstance(model, Pipeline) and "model" in model.named_steps:
        scaler = model.named_steps["scale"]
        linear = model.named_steps["model"]
        explainer = shap.LinearExplainer(linear, scaler.transform(X_bg))
        sv = explainer(scaler.transform(X_pred))
    elif hasattr(model, "model") and hasattr(model, "scaler"):
        X_bg_s = model.scaler.transform(X_bg)
        X_pred_s = model.scaler.transform(X_pred)
        explainer = shap.KernelExplainer(model.model.predict, X_bg_s[:100])
        sv = explainer.shap_values(X_pred_s, nsamples=100)
        sv = shap.Explanation(
            values=np.array(sv).reshape(1, -1),
            base_values=explainer.expected_value,
            data=X_pred_s[0],
        )
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_pred)

    fig, _ = plt.subplots(figsize=(8, 4))
    shap.plots.bar(sv[0] if hasattr(sv, "__len__") and len(sv.shape) > 1 else sv,
                   max_display=12, show=False)
    st.pyplot(fig)
    plt.close(fig)


def render_lime(model, X_bg, X_pred, feature_names):
    if hasattr(model, "aqi_col_idx"):
        st.info("LIME not needed — persistence uses current AQI only.")
        return
    from lime.lime_tabular import LimeTabularExplainer

    def predict_fn(X):
        return model.predict(X.astype(np.float64))

    explainer = LimeTabularExplainer(
        X_bg, feature_names=feature_names, mode="regression", verbose=False,
    )
    exp = explainer.explain_instance(
        X_pred[0], predict_fn, num_features=min(12, len(feature_names)),
    )
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)
    plt.close(fig)


# ---- Header ----
st.title("Pearls AQI Predictor — Lahore")
st.caption("3-day Air Quality Index forecast · serverless ML pipeline")

try:
    registry = load_registry()
except FileNotFoundError:
    st.error("No trained model found. Run `python backfill.py 365` then `python train.py`.")
    st.stop()

try:
    if API_URL:
        api_forecast = fetch_from_api("/forecast")
        current_aqi = api_forecast["current"]["aqi"]
        label = api_forecast["current"]["category"]
        forecast_points = api_forecast["forecast"]
        peak_aqi = api_forecast["peak_aqi"]
        last_updated = api_forecast.get("last_updated")
        data_sources = api_forecast.get("data_sources", {})
        feats = get_recent_features()
    else:
        feats = get_recent_features()
        latest = feats.iloc[-1]
        fc = full_forecast(latest, registry)
        current_aqi = fc["current"]["aqi"]
        label = fc["current"]["category"]
        forecast_points = fc["forecast"]
        peak_aqi = fc["peak_aqi"]
        last_updated = str(latest_timestamp(feats))
        data_sources = dict(fp.LAST_DATA_SOURCES)
except Exception as e:
    st.error(f"Could not fetch live data: {e}")
    st.stop()

h72_info = registry["by_horizon"]["72"]
h72_cols = horizon_feature_cols(registry, 72)
latest = feats.iloc[-1]

src_weather = data_sources.get("weather", "unknown")
src_pollutants = data_sources.get("pollutants", "unknown")
feature_store = os.getenv("FEATURE_STORE", "local")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current AQI", current_aqi)
c2.markdown(f"**{label}**")
c3.metric("Best model (72h)", h72_info["best_model"])
c4.metric("Data points", len(feats))

st.caption(
    f"Last updated: {last_updated} · Weather: {src_weather} · "
    f"Pollutants: {src_pollutants} · Feature store: {feature_store}"
    + (f" · API: {API_URL}" if API_URL else "")
)

# ---- Alerts ----
if current_aqi > 300:
    st.error("HAZARDOUS air quality — avoid outdoor activity.")
elif current_aqi > 200:
    st.warning("Very unhealthy — limit outdoor exposure.")
elif current_aqi > 150:
    st.warning("Unhealthy for sensitive groups.")

# ---- 3-Day Forecast ----
st.subheader("3-Day Forecast")
fdf = pd.DataFrame(forecast_points)
if "hour_ahead" in fdf.columns:
    chart_cols = [c for c in ["forecast_aqi", "lower", "upper"] if c in fdf.columns]
    if chart_cols:
        st.line_chart(fdf.set_index("hour_ahead")[chart_cols])

pts = {p["hour_ahead"]: p for p in forecast_points}
fc1, fc2, fc3 = st.columns(3)
fc1.metric("24h forecast", int(pts.get(24, {}).get("forecast_aqi", 0)))
fc2.metric("48h forecast", int(pts.get(48, {}).get("forecast_aqi", 0)))
fc3.metric("72h forecast", int(pts.get(72, {}).get("forecast_aqi", 0)))

if peak_aqi > 300:
    st.error(f"Forecast peak AQI {peak_aqi} may reach HAZARDOUS levels.")
elif peak_aqi > 200:
    st.warning(f"Forecast peak AQI {peak_aqi} may reach very unhealthy levels.")

# ---- Model comparison ----
st.subheader("Model Comparison (72h horizon)")
model_names = [m for m in registry["by_horizon"]["72"]["models"] if m != "persistence"]
selected = st.selectbox(
    "Select model for explanation",
    model_names,
    index=model_names.index(h72_info["best_model"]) if h72_info["best_model"] in model_names else 0,
)

comp_rows = []
for name, info in registry["by_horizon"]["72"]["models"].items():
    comp_rows.append({
        "model": name,
        "RMSE": round(info["rmse"], 2),
        "MAE": round(info["mae"], 2),
        "R²": round(info["r2"], 3),
        "best": name == h72_info["best_model"],
    })
st.dataframe(pd.DataFrame(comp_rows).sort_values("RMSE"), hide_index=True)

explain_model = load_named_model(selected, 72)
X_now = feature_vector(latest, h72_cols)
X_bg = feats[h72_cols].astype(np.float64).values

tab_shap, tab_lime = st.tabs(["SHAP", "LIME"])
with tab_shap:
    st.markdown(f"**SHAP — {selected} @ 72h**")
    try:
        render_shap(explain_model, X_bg, X_now, h72_cols)
    except Exception as e:
        st.info(f"SHAP unavailable: {e}")
with tab_lime:
    st.markdown(f"**LIME — {selected} @ 72h**")
    try:
        render_lime(explain_model, X_bg, X_now, h72_cols)
    except Exception as e:
        st.info(f"LIME unavailable: {e}")

st.divider()

# ---- EDA ----
st.subheader("Exploratory Data Analysis")
eda = feats.copy()
ec1, ec2 = st.columns(2)
with ec1:
    st.markdown("**AQI over recent days**")
    st.line_chart(eda.set_index("time")["aqi"])
with ec2:
    st.markdown("**Average AQI by hour of day**")
    st.bar_chart(eda.groupby("hour")["aqi"].mean())
ec3, ec4 = st.columns(2)
with ec3:
    if "temperature_2m" in eda.columns:
        st.markdown("**Temperature vs AQI**")
        st.scatter_chart(eda, x="temperature_2m", y="aqi")
with ec4:
    st.markdown("**Monthly average AQI**")
    st.bar_chart(eda.groupby("month")["aqi"].mean())

st.divider()
st.subheader("Model Metrics (all horizons)")
metric_rows = []
for h in HORIZONS:
    he = registry["by_horizon"][str(h)]
    metric_rows.append({
        "horizon": f"{h}h",
        "best_model": he["best_model"],
        "RMSE": round(he["best_metrics"]["rmse"], 2),
        "MAE": round(he["best_metrics"]["mae"], 2),
        "R²": round(he["best_metrics"]["r2"], 3),
        "beats_persistence": he.get("persistence_beaten", False),
    })
st.dataframe(pd.DataFrame(metric_rows), hide_index=True)
