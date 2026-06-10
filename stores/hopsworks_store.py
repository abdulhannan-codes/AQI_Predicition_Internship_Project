"""Hopsworks Feature Store + Model Registry."""
import datetime as dt
import os
import pathlib
import tempfile

import joblib
import pandas as pd

from config import FEATURE_COLS, FEATURE_INT_COLS

FG_NAME = "aqi_features"
FG_VERSION = 1

FEATURE_INT_COLS_SET = set(FEATURE_INT_COLS)


class HopsworksStore:
    def __init__(self):
        self._project = None
        self._fg = None

    def _connection(self):
        if self._project is not None:
            return self._project
        import hopsworks
        api_key = os.environ.get("HOPSWORKS_API_KEY")
        if not api_key:
            raise ValueError("HOPSWORKS_API_KEY is not set")
        project_name = os.getenv("HOPSWORKS_PROJECT", "pearls-aqi-predictor")
        try:
            self._project = hopsworks.login(api_key_value=api_key, project=project_name)
        except Exception as exc:
            raise RuntimeError(
                f"Hopsworks login failed for project '{project_name}'. "
                "Create the API key inside your project (Settings → API Keys) with PROJECT scope. "
                f"Original error: {exc}"
            ) from exc
        return self._project

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "time" not in out.columns:
            raise ValueError("Feature dataframe must include 'time' column")
        out["time"] = pd.to_datetime(out["time"])
        cols = ["time"] + [c for c in FEATURE_COLS if c in out.columns]
        out = out[cols].dropna().reset_index(drop=True)
        if out.empty:
            raise ValueError("No complete feature rows to insert into Hopsworks")
        # Match Hopsworks schema: bigint for time-derived ints, double for the rest
        for col in FEATURE_COLS:
            if col not in out.columns:
                continue
            if col in FEATURE_INT_COLS_SET:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("int64")
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        return out

    def _feature_group(self, fs):
        """Return a FeatureGroup handle; never None."""
        if self._fg is not None:
            return self._fg

        fg = None
        try:
            fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
        except Exception:
            fg = None

        if fg is None and hasattr(fs, "get_or_create_feature_group"):
            fg = fs.get_or_create_feature_group(
                name=FG_NAME,
                version=FG_VERSION,
                description="Hourly AQI features for Lahore",
                primary_key=["time"],
                event_time="time",
                online_enabled=False,
            )

        if fg is None:
            fg = fs.create_feature_group(
                name=FG_NAME,
                version=FG_VERSION,
                description="Hourly AQI features for Lahore",
                primary_key=["time"],
                event_time="time",
                online_enabled=False,
            )

        if fg is None:
            raise RuntimeError(
                f"Could not get or create feature group '{FG_NAME}' v{FG_VERSION}. "
                "Check Hopsworks project permissions and API key PROJECT scope."
            )

        self._fg = fg
        return self._fg

    def save(self, df: pd.DataFrame, name: str = "features.parquet") -> pathlib.Path:
        out = self._prepare_frame(df)

        # Always mirror locally first
        local = pathlib.Path(__file__).resolve().parent.parent / "data" / name
        local.parent.mkdir(exist_ok=True)
        out.to_parquet(local, index=False)

        fs = self._connection().get_feature_store()
        fg = self._feature_group(fs)
        if fg is None or not hasattr(fg, "insert"):
            raise RuntimeError(
                f"Invalid feature group object for '{FG_NAME}': {type(fg)} — "
                "expected Hopsworks FeatureGroup with insert()."
            )

        print(f"Hopsworks: inserting {len(out)} rows into {FG_NAME} v{FG_VERSION}")
        fg.insert(out)
        return local

    def load(self, name: str = "training_data.parquet") -> pd.DataFrame:
        fs = self._connection().get_feature_store()
        fg = None
        try:
            fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
        except Exception:
            fg = None

        if fg is not None:
            try:
                df = fg.read()
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            except Exception:
                pass

        local = pathlib.Path(__file__).resolve().parent.parent / "data" / name
        if local.exists():
            return pd.read_parquet(local)
        raise FileNotFoundError("No features in Hopsworks or local store")

    def load_range(self, start: dt.date, end: dt.date, name: str = "training_data.parquet") -> pd.DataFrame:
        df = self.load(name)
        mask = (pd.to_datetime(df["time"]) >= pd.Timestamp(start)) & (
            pd.to_datetime(df["time"]) <= pd.Timestamp(end)
        )
        return df.loc[mask].reset_index(drop=True)


def register_model(
    model,
    model_name: str,
    horizon: int,
    metrics: dict,
    description: str = "",
) -> int | None:
    """Register a trained model in Hopsworks Model Registry. Returns model version."""
    if not os.getenv("HOPSWORKS_API_KEY"):
        return None
    store = HopsworksStore()
    mr = store._connection().get_model_registry()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / f"{model_name}_h{horizon}.pkl"
        joblib.dump(model, path)
        mv = mr.python.create_model(
            name=f"aqi_{model_name}_h{horizon}",
            description=description or f"AQI {horizon}h forecast — RMSE {metrics.get('rmse', 0):.2f}",
            metrics=metrics,
        )
        mv.save(str(path))
        return mv.version
