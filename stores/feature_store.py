"""Feature Store factory — local Parquet or Hopsworks."""
import os

from stores.local_store import LocalParquetStore


def get_feature_store():
    backend = os.getenv("FEATURE_STORE", "local").lower()
    if backend == "hopsworks" and os.getenv("HOPSWORKS_API_KEY"):
        from stores.hopsworks_store import HopsworksStore
        return HopsworksStore()
    return LocalParquetStore()
