# Pearls AQI Predictor — Lahore

Predict Lahore's Air Quality Index (AQI) for the next **3 days** (24h / 48h / 72h)
using an end-to-end ML pipeline with Hopsworks Feature Store, FastAPI backend,
and Streamlit dashboard.

---

## Quick start (local)

```bash
python -m pip install -r requirements.txt
cp .env.example .env          # add OPENWEATHER_API_KEY, HOPSWORKS_API_KEY (optional)

python backfill.py 365
python train.py
uvicorn api:app --reload --port 8000    # terminal 1
streamlit run app.py                    # terminal 2
```

Or both at once: `bash start.sh` (sets `API_URL=http://localhost:8000`).

---

## Architecture

```
OpenWeather + Open-Meteo  →  feature_pipeline.py
                                ↓
                    Feature Store (local Parquet / Hopsworks)
                                ↓
                         train.py (RF, Ridge, TensorFlow)
                                ↓
                    Model Registry (local + Hopsworks)
                                ↓
              api.py (FastAPI)  →  app.py (Streamlit)
```

**CI/CD:** GitHub Actions — hourly features, daily retraining.

**Deploy:** Render (FastAPI) + Streamlit Community Cloud (dashboard with `API_URL`).

---

## Environment variables

| Variable | Purpose |
|---|---|
| `OPENWEATHER_API_KEY` | Primary weather source (spec) |
| `HOPSWORKS_API_KEY` | Hopsworks Feature Store + Model Registry |
| `HOPSWORKS_PROJECT` | Hopsworks project name |
| `FEATURE_STORE` | `local` or `hopsworks` |
| `API_URL` | Streamlit → FastAPI URL (empty = direct services mode) |

---

## API endpoints (FastAPI)

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness |
| `GET /aqi/current` | Current AQI + category |
| `GET /forecast` | 24h / 48h / 72h forecast JSON |
| `GET /models/metrics` | Model registry metrics |
| `GET /features/recent?hours=72` | Recent feature rows |

---

## Full requirements checklist (33/33)

| # | Requirement | Status |
|---|---|---|
| 1 | Python | Done |
| 2 | Scikit-learn | Done |
| 3 | TensorFlow | Done |
| 4 | Hopsworks / Vertex AI Feature Store | Done (Hopsworks) |
| 5 | GitHub Actions / Airflow | Done (GitHub Actions) |
| 6 | Streamlit dashboard | Done |
| 7 | Flask / FastAPI backend | Done (FastAPI) |
| 8 | AQICN / OpenWeather APIs | Done (OpenWeather + Open-Meteo) |
| 9 | SHAP | Done |
| 10 | Git | Done |
| 11 | Fetch weather + pollutant data | Done |
| 12 | Time features (hour, day, month) | Done |
| 13 | Derived features + AQI change rate | Done |
| 14 | Store in Feature Store | Done |
| 15 | Historical backfill | Done |
| 16 | Comprehensive training dataset | Done |
| 17 | Fetch features/targets from Feature Store | Done |
| 18 | Random Forest | Done |
| 19 | Ridge Regression | Done |
| 20 | TensorFlow model | Done |
| 21 | RMSE, MAE, R² | Done |
| 22 | Model Registry | Done |
| 23 | Feature pipeline hourly | Done (GitHub Actions) |
| 24 | Training pipeline daily | Done (GitHub Actions) |
| 25 | Airflow / GitHub Actions | Done |
| 26 | Load models + features from Feature Store | Done |
| 27 | Real-time 3-day predictions | Done |
| 28 | Interactive Streamlit dashboard | Done |
| 29 | Backend serving predictions | Done (FastAPI) |
| 30 | EDA trends | Done |
| 31 | SHAP / LIME explainability | Done |
| 32 | Hazardous AQI alerts | Done |
| 33 | Multiple forecasting models | Done |

---

## Deploy

**Full step-by-step guide:** see [DEPLOYMENT.md](DEPLOYMENT.md)

Quick summary:
1. Push to GitHub: `.\scripts\push_github.ps1` (after `gh auth login`)
2. Fill `.env` with OpenWeather + Hopsworks keys
3. Run `python backfill.py 365` and `python train.py` with `FEATURE_STORE=hopsworks`
4. Add GitHub secrets and trigger Actions workflows
5. Deploy FastAPI on Render, then Streamlit Cloud with `API_URL`

**Verify locally:**
```powershell
.\scripts\verify_checklist.ps1 -ApiUrl "http://localhost:8000"
```

**Public URLs** (fill in after deploy):
- API: `https://YOUR-APP.onrender.com`
- Dashboard: `https://YOUR-APP.streamlit.app`

### Local verification status (last run)

| Check | Status |
|---|---|
| FastAPI `/forecast` | Pass (local) |
| SHAP explainability | Pass |
| Feature pipeline | Pass (`weather=openmeteo` until OpenWeather key set) |
| Hopsworks registry | Pending — add `HOPSWORKS_API_KEY` and re-run `train.py` |
| GitHub Actions | Pending — push repo + add secrets |
| Render / Streamlit deploy | Pending — see [DEPLOYMENT.md](DEPLOYMENT.md) |

### FastAPI on Render
1. Push repo to GitHub
2. Create Web Service from `render.yaml`
3. Set secrets: `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`

### Streamlit Community Cloud
1. Connect GitHub repo, main file `app.py`
2. Add secret: `API_URL=https://your-render-app.onrender.com`

---

## Results (Lahore, 365-day backfill)

| Horizon | Best model | RMSE | R² |
|---|---|---|---|
| 24h | Ridge | 32.2 | 0.28 |
| 48h | Ridge | 39.6 | -0.09 |
| 72h | Persistence | 41.8 | -0.21 |

At 72h, no ML model beat the persistence baseline — dashboard uses current AQI honestly.
