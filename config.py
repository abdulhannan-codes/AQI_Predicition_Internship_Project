"""Shared constants for feature pipeline, training, and dashboard."""

HORIZONS = [24, 48, 72]

FEATURE_COLS = [
    "aqi",
    "hour", "day", "month", "dayofweek", "week_of_year", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone", "sulphur_dioxide",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure", "precipitation",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12", "aqi_lag_24", "aqi_lag_48",
    "pm2_5_lag_1", "pm2_5_lag_2", "pm2_5_lag_3", "pm2_5_lag_6", "pm2_5_lag_12", "pm2_5_lag_24",
    "aqi_roll_mean_6", "aqi_roll_mean_12", "aqi_roll_mean_24", "aqi_roll_mean_48",
    "aqi_roll_std_6", "aqi_roll_std_12", "aqi_roll_std_24", "aqi_roll_std_48",
    "aqi_change_rate", "pm25_change_rate", "temp_humidity", "pm25_pm10_ratio",
]

# Hopsworks dtype mapping (must match existing feature group schema)
FEATURE_INT32_COLS = ["hour", "day", "month", "dayofweek"]
FEATURE_INT64_COLS = ["week_of_year", "is_weekend"]  # bigint in Hopsworks
