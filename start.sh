#!/usr/bin/env bash
# Run API + dashboard locally
uvicorn api:app --host 0.0.0.0 --port 8000 &
sleep 2
export API_URL=http://localhost:8000
streamlit run app.py
