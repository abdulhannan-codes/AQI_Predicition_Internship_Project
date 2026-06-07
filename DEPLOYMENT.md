# Pearls AQI — Deployment Guide

Follow these steps after the local code is working. Steps 3–8 require your accounts.

---

## Step 2 — Push to GitHub

Git is initialized and committed locally. To push:

```powershell
cd d:\pearls-aqi-predictor
gh auth login                    # one-time: follow browser prompts
gh repo create pearls-aqi-predictor --public --source=. --remote=origin --push
```

Or if the repo already exists on GitHub:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/pearls-aqi-predictor.git
git push -u origin main
```

---

## Step 3 — API keys

1. **OpenWeather:** https://openweathermap.org/api → free API key
2. **Hopsworks:** https://www.hopsworks.ai/ → create project `pearls-aqi-predictor` → API keys

Edit `.env`:

```
OPENWEATHER_API_KEY=your_key_here
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT=pearls-aqi-predictor
FEATURE_STORE=hopsworks
```

---

## Step 4 — Prove Hopsworks locally

```powershell
.venv\Scripts\activate
python feature_pipeline.py       # expect: sources: weather=openweather (or openmeteo fallback)
python backfill.py 365
python train.py
```

Verify:
- Hopsworks UI → Feature Group `aqi_features` has rows
- Hopsworks UI → Model Registry has `aqi_*_h24/h48/h72`
- `models/registry.json` contains `"hopsworks_version"` per horizon

---

## Step 5 — GitHub Actions secrets

In GitHub repo → **Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|---|---|
| `HOPSWORKS_API_KEY` | Your Hopsworks API key |
| `HOPSWORKS_PROJECT` | `pearls-aqi-predictor` |
| `OPENWEATHER_API_KEY` | Your OpenWeather key |

Manually trigger both workflows under **Actions** tab:
- Feature Pipeline (hourly)
- Training Pipeline (daily)

---

## Step 6 — Deploy FastAPI on Render

1. Sign up at https://render.com
2. **New → Blueprint** (uses `render.yaml`) or **Web Service**
3. Connect your GitHub repo
4. Set env vars: `OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, `FEATURE_STORE=hopsworks`
5. Test:
   ```
   curl https://YOUR-APP.onrender.com/health
   curl https://YOUR-APP.onrender.com/forecast
   ```

---

## Step 7 — Deploy Streamlit Cloud

1. Sign up at https://share.streamlit.io
2. Connect same GitHub repo, main file: `app.py`
3. Add secret: `API_URL=https://YOUR-APP.onrender.com`
4. Open the Streamlit URL and confirm forecasts load

---

## Step 8 — Final verification

Run the checklist script:

```powershell
.\scripts\verify_checklist.ps1 -ApiUrl "https://YOUR-APP.onrender.com" -StreamlitUrl "https://YOUR-APP.streamlit.app"
```

Then add your public URLs to `README.md`.
