"""Load recent features for inference."""
import datetime as dt

import pandas as pd

import feature_pipeline as fp
from config import FEATURE_COLS
from stores.feature_store import get_feature_store

# Set by feature_pipeline after fetch
LAST_DATA_SOURCES: dict[str, str] = {"weather": "unknown", "pollutants": "unknown"}


def get_recent_features(days_back: int = 60) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    df = None

    try:
        df = fp.build_features(start, end)
    except Exception:
        df = None

    missing = [c for c in FEATURE_COLS if c not in (df.columns if df is not None else [])]
    if df is None or missing:
        store = get_feature_store()
        try:
            df = store.load("training_data.parquet")
            cutoff = pd.Timestamp(start)
            df = df[df["time"] >= cutoff].copy()
            if df.empty:
                df = store.load("training_data.parquet").tail(1500).copy()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Could not load features. Run: python backfill.py 365"
            ) from exc

    cols = [c for c in FEATURE_COLS if c in df.columns]
    still_missing = [c for c in FEATURE_COLS if c not in df.columns]
    if still_missing:
        raise RuntimeError(f"Missing feature columns: {still_missing}")

    df = df.dropna(subset=cols).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No complete feature rows after dropna.")
    return df


def latest_timestamp(feats: pd.DataFrame):
    if feats.empty or "time" not in feats.columns:
        return None
    return pd.to_datetime(feats["time"].iloc[-1])
