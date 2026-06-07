"""Local Parquet Feature Store (development / fallback)."""
import datetime as dt
import pathlib

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class LocalParquetStore:
    def save(self, df: pd.DataFrame, name: str = "features.parquet") -> pathlib.Path:
        out = DATA_DIR / name
        df.to_parquet(out, index=False)
        return out

    def load(self, name: str = "features.parquet") -> pd.DataFrame:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        return pd.read_parquet(path)

    def load_range(self, start: dt.date, end: dt.date, name: str = "training_data.parquet") -> pd.DataFrame:
        df = self.load(name)
        mask = (df["time"] >= pd.Timestamp(start)) & (df["time"] <= pd.Timestamp(end))
        return df.loc[mask].reset_index(drop=True)
