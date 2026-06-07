"""
feature_pipeline.py
Fetch raw pollutant + weather data for Lahore and compute features.

Requirement mapping (Pearls spec -> code):
  - "Fetch raw weather and pollutant data from external APIs"  -> fetch_pollutants(), fetch_weather()
  - "Compute time-based features (hour, day, month)"           -> add_time_features()
  - "Derived features like AQI change rate"                    -> add_derived_features()
  - "Store processed features in Feature Store"                -> save_features() / load_features()
"""
from env_loader import load_env

load_env()

import datetime as dt
import os
import pathlib

import numpy as np
import pandas as pd
import requests

LAHORE_LAT = float(os.getenv("LAT", "31.5497"))
LAHORE_LON = float(os.getenv("LON", "74.3436"))
DATA_DIR = pathlib.Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_TIMEMACHINE = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

# Updated by fetch_raw — read by dashboard
LAST_DATA_SOURCES: dict[str, str] = {"weather": "openmeteo", "pollutants": "openmeteo"}

POLLUTANTS = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone", "sulphur_dioxide",
]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "surface_pressure", "precipitation",
]


def fetch_pollutants(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Hourly pollutant concentrations from Open-Meteo air-quality API."""
    params = {
        "latitude": LAHORE_LAT,
        "longitude": LAHORE_LON,
        "hourly": ",".join(POLLUTANTS),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "Asia/Karachi",
    }
    r = requests.get(AQ_URL, params=params, timeout=90)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def fetch_weather_openweather_history(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Hourly weather from OpenWeather One Call Timemachine (spec: OpenWeather API)."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return pd.DataFrame(columns=["time"] + WEATHER_VARS)

    frames = []
    day = start
    while day <= end:
        try:
            params = {
                "lat": LAHORE_LAT,
                "lon": LAHORE_LON,
                "dt": int(dt.datetime.combine(day, dt.time.min).timestamp()),
                "appid": key,
                "units": "metric",
            }
            r = requests.get(OPENWEATHER_TIMEMACHINE, params=params, timeout=60)
            if r.status_code == 401:
                break
            r.raise_for_status()
            data = r.json()
            hourly = data.get("data", [])
            rows = []
            for h in hourly:
                ts = pd.to_datetime(h["dt"], unit="s")
                if start <= ts.date() <= end:
                    rows.append({
                        "time": ts,
                        "temperature_2m": h.get("temp"),
                        "relative_humidity_2m": h.get("humidity"),
                        "wind_speed_10m": h.get("wind_speed"),
                        "surface_pressure": h.get("pressure"),
                        "precipitation": h.get("rain", {}).get("1h", 0.0),
                    })
            if rows:
                frames.append(pd.DataFrame(rows))
        except requests.RequestException:
            pass
        day += dt.timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["time"] + WEATHER_VARS)
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"]).dt.floor("h")
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def fetch_weather_openmeteo(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Hourly weather from Open-Meteo archive + forecast for recent days."""
    today = dt.date.today()
    archive_end = min(end, today - dt.timedelta(days=1))
    frames = []

    try:
        if start <= archive_end:
            params = {
                "latitude": LAHORE_LAT,
                "longitude": LAHORE_LON,
                "hourly": ",".join(WEATHER_VARS),
                "start_date": start.isoformat(),
                "end_date": archive_end.isoformat(),
                "timezone": "Asia/Karachi",
            }
            r = requests.get(WEATHER_URL, params=params, timeout=90)
            r.raise_for_status()
            hourly = r.json()["hourly"]
            frames.append(pd.DataFrame(hourly))

        if end > archive_end:
            params = {
                "latitude": LAHORE_LAT,
                "longitude": LAHORE_LON,
                "hourly": ",".join(WEATHER_VARS),
                "timezone": "Asia/Karachi",
                "forecast_days": min(16, (end - archive_end).days + 1),
            }
            r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=90)
            r.raise_for_status()
            hourly = r.json()["hourly"]
            fc = pd.DataFrame(hourly)
            fc["time"] = pd.to_datetime(fc["time"])
            fc = fc[(fc["time"].dt.date >= (archive_end + dt.timedelta(days=1))) &
                    (fc["time"].dt.date <= end)]
            frames.append(fc)
    except requests.RequestException:
        pass

    if not frames:
        return pd.DataFrame(columns=["time"] + WEATHER_VARS)

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def fetch_weather_openweather() -> pd.DataFrame | None:
    """Optional live weather point from OpenWeather when API key is set."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return None
    params = {"lat": LAHORE_LAT, "lon": LAHORE_LON, "appid": key, "units": "metric"}
    r = requests.get(OPENWEATHER_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    now = pd.Timestamp.now(tz="Asia/Karachi").floor("h").tz_localize(None)
    return pd.DataFrame([{
        "time": now,
        "temperature_2m": data["main"]["temp"],
        "relative_humidity_2m": data["main"]["humidity"],
        "wind_speed_10m": data["wind"]["speed"],
        "surface_pressure": data["main"]["pressure"],
        "precipitation": data.get("rain", {}).get("1h", 0.0),
    }])


def fetch_raw(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Merge pollutant + weather hourly data on timestamp."""
    global LAST_DATA_SOURCES

    try:
        aq = fetch_pollutants(start, end)
        LAST_DATA_SOURCES["pollutants"] = "openmeteo"
    except requests.RequestException:
        aq = pd.DataFrame(columns=["time"] + POLLUTANTS)
        LAST_DATA_SOURCES["pollutants"] = "none"

    wx_ow = fetch_weather_openweather_history(start, end)
    if not wx_ow.empty:
        wx = wx_ow
        LAST_DATA_SOURCES["weather"] = "openweather"
    else:
        wx = fetch_weather_openmeteo(start, end)
        LAST_DATA_SOURCES["weather"] = "openmeteo"

    df = aq.merge(wx, on="time", how="outer").sort_values("time").reset_index(drop=True)

    for col in WEATHER_VARS:
        if col not in df.columns:
            df[col] = np.nan

    ow = fetch_weather_openweather()
    if ow is not None and not ow.empty:
        try:
            t = ow["time"].iloc[0]
            mask = df["time"] == t
            if mask.any():
                for col in WEATHER_VARS:
                    df.loc[mask, col] = ow[col].iloc[0]
            else:
                df = pd.concat([df, ow], ignore_index=True).sort_values("time").reset_index(drop=True)
        except requests.RequestException:
            pass

    for col in POLLUTANTS + WEATHER_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.ffill(limit=3).bfill(limit=3)
    return df


def compute_aqi_from_pm25(pm25: pd.Series) -> pd.Series:
    """US EPA AQI from PM2.5 concentration (ug/m3). Piecewise-linear breakpoints."""
    bp = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
    ]

    def one(c):
        if pd.isna(c):
            return np.nan
        c = max(0.0, min(float(c), 500.4))
        for clo, chi, ilo, ihi in bp:
            if clo <= c <= chi:
                return round((ihi - ilo) / (chi - clo) * (c - clo) + ilo)
        return 500

    return pm25.apply(one)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df["time"]
    df["hour"] = t.dt.hour
    df["day"] = t.dt.day
    df["month"] = t.dt.month
    df["dayofweek"] = t.dt.dayofweek
    df["week_of_year"] = t.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags, rolling stats, AQI change rate, and weather-derived features."""
    df = df.copy()
    # EPA PM2.5 AQI uses a 24-hour average concentration
    pm25_24h = df["pm2_5"].rolling(24, min_periods=12).mean()
    df["aqi"] = compute_aqi_from_pm25(pm25_24h)

    for lag in (1, 2, 3, 6, 12, 24, 48):
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
        df[f"pm2_5_lag_{lag}"] = df["pm2_5"].shift(lag)

    for window in (6, 12, 24, 48):
        df[f"aqi_roll_mean_{window}"] = df["aqi"].rolling(window, min_periods=1).mean()
        df[f"aqi_roll_std_{window}"] = df["aqi"].rolling(window, min_periods=1).std()

    df["aqi_change_rate"] = df["aqi"].diff()
    df["pm25_change_rate"] = df["pm2_5"].diff()

    if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        df["temp_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100.0

    if "pm10" in df.columns:
        df["pm25_pm10_ratio"] = df["pm2_5"] / df["pm10"].replace(0, np.nan)

    return df


def build_features(start: dt.date, end: dt.date) -> pd.DataFrame:
    from config import FEATURE_COLS

    raw = fetch_raw(start, end)
    df = add_derived_features(add_time_features(raw))
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def save_features(df: pd.DataFrame, name: str = "features.parquet") -> pathlib.Path:
    """Persist to Feature Store (Hopsworks or local Parquet)."""
    from stores.feature_store import get_feature_store
    return get_feature_store().save(df, name=name)


def load_features(name: str = "features.parquet") -> pd.DataFrame:
    """Load from Feature Store."""
    from stores.feature_store import get_feature_store
    return get_feature_store().load(name=name)


if __name__ == "__main__":
    end = dt.date.today()
    start = end - dt.timedelta(days=7)
    df = build_features(start, end)
    from config import FEATURE_COLS
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=cols).reset_index(drop=True)
    p = save_features(df)
    print(f"Wrote {len(df)} rows -> {p}")
    print(f"sources: weather={LAST_DATA_SOURCES['weather']}, pollutants={LAST_DATA_SOURCES['pollutants']}")
    print(df[["time", "aqi", "pm2_5", "temperature_2m"]].tail(3).to_string())
