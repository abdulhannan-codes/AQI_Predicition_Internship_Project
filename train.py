"""
train.py
Train and compare models for 24h / 48h / 72h AQI forecasts.

Requirement mapping:
  - Random Forest, Ridge Regression, TensorFlow models
  - RMSE, MAE, R² evaluation
  - Model Registry -> models/ + registry.json
"""
import datetime as dt
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLS, HORIZONS

DATA_DIR = pathlib.Path(__file__).parent / "data"
MODELS_DIR = pathlib.Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def sklearn_model_templates():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ]),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        ),
    }


from models_def import PersistenceModel, TensorFlowAQIModel


def persistence_predict(current_aqi: np.ndarray) -> np.ndarray:
    """Baseline: future AQI equals current AQI."""
    return current_aqi.copy()


def load_training_data() -> pd.DataFrame:
    from stores.feature_store import get_feature_store
    store = get_feature_store()
    try:
        return store.load("training_data.parquet")
    except FileNotFoundError:
        path = DATA_DIR / "training_data.parquet"
        if not path.exists():
            raise FileNotFoundError("Run backfill.py first to create training_data.parquet")
        return pd.read_parquet(path)


def make_supervised(df: pd.DataFrame, horizon: int):
    df = df.sort_values("time").reset_index(drop=True).copy()
    df["target"] = df["aqi"].shift(-horizon)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=cols + ["target", "aqi"])
    X = df[cols].astype(np.float64).values
    y = df["target"].astype(np.float64).values
    current_aqi = df["aqi"].astype(np.float64).values
    return X, y, current_aqi, cols


def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }



def train_all():
    df = load_training_data()
    registry = {
        "trained_at": dt.datetime.now().isoformat(),
        "horizons": HORIZONS,
        "features": FEATURE_COLS,
        "by_horizon": {},
    }

    overall_best = None

    for horizon in HORIZONS:
        X, y, current_aqi, cols = make_supervised(df, horizon)
        split = int(len(X) * 0.8)
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]
        aqi_te = current_aqi[split:]

        print(f"\n=== Horizon {horizon}h ({len(Xtr)} train / {len(Xte)} test) ===")

        horizon_entry = {
            "horizon_hours": horizon,
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "feature_cols": cols,
            "models": {},
        }

        # Persistence baseline
        base_metrics = evaluate(yte, persistence_predict(aqi_te))
        horizon_entry["models"]["persistence"] = {"path": None, **base_metrics}
        print(f"{'persistence':20s}  RMSE={base_metrics['rmse']:6.2f}  "
              f"MAE={base_metrics['mae']:6.2f}  R2={base_metrics['r2']:.3f}")

        results = []

        for name, template in sklearn_model_templates().items():
            model = clone(template)
            model.fit(Xtr, ytr)
            metrics = evaluate(yte, model.predict(Xte))
            path = MODELS_DIR / f"{name}_h{horizon}.pkl"
            joblib.dump(model, path)
            horizon_entry["models"][name] = {"path": path.name, **metrics}
            results.append((name, metrics, model))
            print(f"{name:20s}  RMSE={metrics['rmse']:6.2f}  "
                  f"MAE={metrics['mae']:6.2f}  R2={metrics['r2']:.3f}")

        # TensorFlow (spec requirement)
        tf_model = TensorFlowAQIModel()
        tf_model.fit(Xtr, ytr)
        tf_pred = tf_model.predict(Xte)
        tf_metrics = evaluate(yte, tf_pred)
        tf_path = MODELS_DIR / f"tensorflow_h{horizon}.pkl"
        joblib.dump(tf_model, tf_path)
        horizon_entry["models"]["tensorflow"] = {"path": tf_path.name, **tf_metrics}
        results.append(("tensorflow", tf_metrics, tf_model))
        print(f"{'tensorflow':20s}  RMSE={tf_metrics['rmse']:6.2f}  "
              f"MAE={tf_metrics['mae']:6.2f}  R2={tf_metrics['r2']:.3f}")

        # Best ML model must beat persistence; otherwise use persistence
        ml_results = [(n, m, mod) for n, m, mod in results if m["rmse"] < base_metrics["rmse"]]
        if ml_results:
            best_name, best_metrics, best_model = min(ml_results, key=lambda r: r[1]["rmse"])
        else:
            aqi_idx = cols.index("aqi")
            best_name = "persistence"
            best_metrics = base_metrics
            best_model = PersistenceModel(aqi_idx)

        horizon_entry["best_model"] = best_name
        horizon_entry["best_metrics"] = best_metrics
        horizon_entry["persistence_beaten"] = best_metrics["rmse"] < base_metrics["rmse"]

        best_path = MODELS_DIR / f"best_model_h{horizon}.pkl"
        joblib.dump(best_model, best_path)
        horizon_entry["best_model_path"] = best_path.name

        # Hopsworks Model Registry (#22)
        try:
            from stores.hopsworks_store import register_model
            hw_ver = register_model(
                best_model,
                best_name,
                horizon,
                best_metrics,
                description=f"Best AQI model at {horizon}h horizon",
            )
            if hw_ver is not None:
                horizon_entry["hopsworks_version"] = hw_ver
        except Exception as exc:
            print(f"Hopsworks registry skip: {exc}")

        registry["by_horizon"][str(horizon)] = horizon_entry
        print(f"Best at {horizon}h: {best_name}  (beats persistence: "
              f"{horizon_entry['persistence_beaten']})")

        if overall_best is None or best_metrics["rmse"] < overall_best[2]["rmse"]:
            overall_best = (horizon, best_name, best_metrics, best_model)

    # Default served model: best at 72h (3-day forecast)
    h72 = registry["by_horizon"]["72"]
    default_model = joblib.load(MODELS_DIR / h72["best_model_path"])
    joblib.dump(default_model, MODELS_DIR / "best_model.pkl")
    registry["best_model"] = h72["best_model"]
    registry["best_model_horizon"] = 72

    (MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=2))
    print(f"\nRegistry written. Default model: {registry['best_model']} @ 72h")
    return registry


if __name__ == "__main__":
    train_all()
