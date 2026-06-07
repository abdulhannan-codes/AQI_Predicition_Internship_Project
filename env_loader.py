"""Load .env and Streamlit Cloud secrets into os.environ."""
import os
import pathlib

_SECRET_KEYS = (
    "OPENWEATHER_API_KEY",
    "HOPSWORKS_API_KEY",
    "HOPSWORKS_PROJECT",
    "FEATURE_STORE",
    "API_URL",
    "AQICN_TOKEN",
    "CITY",
    "LAT",
    "LON",
)


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = pathlib.Path(__file__).resolve().parent
    load_dotenv(root / ".env")


def apply_streamlit_secrets() -> None:
    """Map Streamlit Cloud secrets → env vars (for feature_pipeline / services)."""
    try:
        import streamlit as st
    except ImportError:
        return
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in _SECRET_KEYS:
        if key in secrets and not os.getenv(key):
            os.environ[key] = str(secrets[key])
