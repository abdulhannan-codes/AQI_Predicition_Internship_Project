"""
backfill.py
Run the feature pipeline over a long past window to build the training dataset.
"""
import datetime as dt

import feature_pipeline as fp
from config import FEATURE_COLS


def backfill(days_back: int = 365) -> None:
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    print(f"Backfilling {start} -> {end} ({days_back} days)...")
    df = fp.build_features(start, end)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=cols + ["aqi"]).reset_index(drop=True)
    path = fp.save_features(df, name="training_data.parquet")
    print(f"Training dataset: {len(df)} rows, {len(cols)} feature cols -> {path}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    backfill(n)
