import sys
import time
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import ReplaceOne

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import (
    DEFAULT_LAT,
    DEFAULT_LON,
    MONGO_DB_NAME,
    MONGO_FEATURES_COLLECTION,
    MONGO_URI,
)
from src.data_collection.fetch_data import fetch_historical_aqi, fetch_openmeteo_historical_aqi, process_raw_data
from src.schema import FEATURE_COLUMNS

load_dotenv()

print("Starting Feature Engineering...")


def fetch_historical_aqi_chunks(start_time: int, end_time: int, chunk_days: int = 30) -> list[dict]:
    """Fetch OpenWeather history in chunks to avoid large-response timeouts."""
    chunk_seconds = chunk_days * 24 * 60 * 60
    rows: list[dict] = []
    current_start = start_time
    while current_start < end_time:
        current_end = min(end_time, current_start + chunk_seconds)
        chunk = fetch_historical_aqi(DEFAULT_LAT, DEFAULT_LON, current_start, current_end)
        source = "OpenWeather"
        if not chunk:
            print("OpenWeather chunk failed or returned no data; trying Open-Meteo fallback.")
            chunk = fetch_openmeteo_historical_aqi(DEFAULT_LAT, DEFAULT_LON, current_start, current_end)
            source = "Open-Meteo"
        if chunk:
            rows.extend(chunk)
            print(f"Fetched {len(chunk)} {source} rows; running total: {len(rows)}")
        else:
            print(f"No data returned for chunk {current_start} -> {current_end}")
        current_start = current_end + 1
    return rows


def build_daily_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Create a daily aggregate dataset for historical charts and sanity checks."""
    daily = hourly_df.copy()
    daily["date"] = daily["timestamp"].dt.date
    aqi_col = "us_aqi" if "us_aqi" in daily.columns else "aqi"
    grouped = (
        daily.groupby("date", as_index=False)
        .agg(
            us_aqi=(aqi_col, "mean"),
            us_aqi_max=(aqi_col, "max"),
            us_aqi_std=(aqi_col, "std"),
            pm2_5=("pm2_5", "mean"),
            pm10=("pm10", "mean"),
            co=("co", "mean"),
            no2=("no2", "mean"),
            o3=("o3", "mean"),
            aqi=("aqi", "mean"),
        )
    )
    grouped["date"] = pd.to_datetime(grouped["date"])
    grouped["day_of_week"] = grouped["date"].dt.dayofweek
    grouped["month"] = grouped["date"].dt.month
    grouped["is_weekend"] = grouped["day_of_week"].isin([5, 6]).astype(int)
    grouped["aqi_lag_1d"] = grouped["us_aqi"].shift(1)
    grouped["aqi_lag_3d"] = grouped["us_aqi"].shift(3)
    grouped["aqi_lag_7d"] = grouped["us_aqi"].shift(7)
    grouped["aqi_rolling_3d"] = grouped["us_aqi"].rolling(window=3).mean()
    grouped["aqi_rolling_7d"] = grouped["us_aqi"].rolling(window=7).mean()
    grouped["pm25_rolling_7d"] = grouped["pm2_5"].rolling(window=7).mean()
    grouped["pm10_rolling_7d"] = grouped["pm10"].rolling(window=7).mean()
    grouped["co_rolling_7d"] = grouped["co"].rolling(window=7).mean()
    grouped["aqi_change_1d"] = grouped["us_aqi"].diff(1)
    grouped["aqi_change_3d"] = grouped["us_aqi"].diff(3)
    return grouped.dropna().reset_index(drop=True)


def _build_feature_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

    if "us_aqi" not in dataframe.columns:
        if "pm2_5" in dataframe.columns:
            from src.data_collection.fetch_data import pm25_to_us_aqi

            dataframe["us_aqi"] = dataframe["pm2_5"].apply(pm25_to_us_aqi)
        else:
            dataframe["us_aqi"] = dataframe["aqi"].astype(float) * 50.0

    dataframe["aqi_raw"] = dataframe["aqi"].astype(float)
    dataframe["aqi"] = dataframe["us_aqi"].astype(float)

    dataframe["hour"] = dataframe["timestamp"].dt.hour
    dataframe["day_of_week"] = dataframe["timestamp"].dt.dayofweek
    dataframe["month"] = dataframe["timestamp"].dt.month
    dataframe["is_weekend"] = dataframe["day_of_week"].isin([5, 6]).astype(int)

    aqi_series = dataframe["us_aqi"].astype(float)
    dataframe["aqi_lag_1"] = aqi_series.shift(1)
    dataframe["aqi_lag_24"] = aqi_series.shift(24)
    dataframe["aqi_lag_48"] = aqi_series.shift(48)
    dataframe["aqi_lag_72"] = aqi_series.shift(72)
    dataframe["aqi_rolling_24"] = aqi_series.rolling(window=24).mean()
    dataframe["pm25_rolling_24"] = dataframe["pm2_5"].rolling(window=24).mean()
    dataframe["pm10_rolling_24"] = dataframe["pm10"].rolling(window=24).mean()
    dataframe["co_rolling_24"] = dataframe["co"].rolling(window=24).mean()
    dataframe["aqi_change_rate"] = aqi_series.pct_change().replace([float("inf"), float("-inf")], pd.NA)

    return dataframe.dropna().reset_index(drop=True)


def build_features() -> pd.DataFrame:
    end_time = int(time.time())
    start_time = end_time - (2 * 365 * 24 * 60 * 60)
    raw_data = fetch_historical_aqi_chunks(start_time, end_time)
    if not raw_data:
        raise RuntimeError("No OpenWeather data was returned.")

    dataframe = _build_feature_dataframe(process_raw_data(raw_data))

    print("Connecting to MongoDB feature store...")
    from pymongo import MongoClient

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    collection = client[MONGO_DB_NAME][MONGO_FEATURES_COLLECTION]
    upload_columns = ["timestamp"] + FEATURE_COLUMNS
    upload_df = dataframe[[column for column in upload_columns if column in dataframe.columns]].copy()
    records = json.loads(upload_df.to_json(orient="records", date_format="iso"))
    operations = [ReplaceOne({"timestamp": record["timestamp"]}, record, upsert=True) for record in records]
    if operations:
        collection.bulk_write(operations, ordered=False)
    collection.create_index("timestamp", unique=True)
    print(f"Successfully upserted {len(records)} feature rows into MongoDB!")

    print("Feature Engineering Complete!")
    print(f"Final dataset shape: {dataframe.shape}")
    print("New features created and stored in the MongoDB feature store.")
    return dataframe


if __name__ == "__main__":
    build_features()
