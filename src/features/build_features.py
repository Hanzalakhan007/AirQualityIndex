import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

load_dotenv()

print("Starting Feature Engineering...")


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


def build_features(input_path: str, output_path: str) -> None:
    dataframe = pd.read_csv(input_path)
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
    dataframe["aqi_rolling_24"] = aqi_series.rolling(window=24).mean()
    dataframe["pm25_rolling_24"] = dataframe["pm2_5"].rolling(window=24).mean()
    dataframe["pm10_rolling_24"] = dataframe["pm10"].rolling(window=24).mean()
    dataframe["co_rolling_24"] = dataframe["co"].rolling(window=24).mean()
    dataframe["aqi_change_rate"] = aqi_series.pct_change().replace([float("inf"), float("-inf")], pd.NA)

    dataframe = dataframe.dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    daily_output_path = Path(output_path).with_name("daily_features.csv")
    build_daily_features(dataframe).to_csv(daily_output_path, index=False)

    print("Connecting to Hopsworks Feature Store...")
    try:
        import hopsworks

        project = hopsworks.login()
        feature_store = project.get_feature_store()
        print(f"Uploading features to Feature Group '{FEATURE_GROUP_NAME}'...")
        feature_group = feature_store.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["timestamp"],
            description="AQI dataset with engineered hourly features",
        )
        upload_columns = [
            "timestamp",
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
            "aqi_rolling_24",
        ]
        upload_df = dataframe[[column for column in upload_columns if column in dataframe.columns]].copy()
        feature_group.insert(upload_df)
        print("Successfully uploaded features to Hopsworks!")
    except Exception as exc:
        print(f"Skipping Hopsworks upload: {exc}")

    print("Feature Engineering Complete!")
    print(f"Final dataset shape: {dataframe.shape}")
    print("New features created: us_aqi, lagged AQI, rolling pollutant features, and AQI change rate")
    print(f"Saved ML-ready data to: {output_path}")


if __name__ == "__main__":
    build_features("data/raw/historical_aqi.csv", "data/processed/features.csv")
