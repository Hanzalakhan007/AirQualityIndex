"""Shared feature schema for training and inference."""

FEATURE_COLUMNS = [
    "aqi",
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "aqi_lag_1",
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_lag_72",
    "aqi_rolling_24",
]
