import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import (
    DEFAULT_LAT,
    DEFAULT_LON,
    OPENWEATHER_AIR_POLLUTION_HISTORY,
    OPENWEATHER_API_KEY,
)

load_dotenv()


def pm25_to_us_aqi(pm25_ugm3: float | None) -> float | None:
    """Convert PM2.5 concentration to US EPA AQI."""
    if pm25_ugm3 is None or pd.isna(pm25_ugm3):
        return None
    value = float(pm25_ugm3)
    if value < 0:
        return None

    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 400),
        (325.5, 425.4, 401, 500),
    ]
    for low_cp, high_cp, low_aqi, high_aqi in breakpoints:
        if low_cp <= value <= high_cp:
            return round(((high_aqi - low_aqi) / (high_cp - low_cp)) * (value - low_cp) + low_aqi, 1)
    return 500.0


def fetch_historical_aqi(lat: float, lon: float, start_date: int, end_date: int) -> list[dict] | None:
    """Fetch historical pollutant measurements from OpenWeather."""
    print(f"Fetching data for Lat: {lat}, Lon: {lon}...")
    params = {
        "lat": lat,
        "lon": lon,
        "start": start_date,
        "end": end_date,
        "appid": OPENWEATHER_API_KEY,
    }
    response = requests.get(OPENWEATHER_AIR_POLLUTION_HISTORY, params=params, timeout=30)
    if response.status_code == 200:
        return response.json().get("list", [])
    print(f"Error fetching data: {response.status_code}")
    print(response.text)
    return None


def process_and_save_data(raw_data: list[dict], output_path: str) -> None:
    """Process raw OpenWeather AQI data into a training-ready CSV."""
    if not raw_data:
        print("No data to process.")
        return

    rows = []
    for item in raw_data:
        timestamp = datetime.fromtimestamp(item["dt"])
        aqi = item.get("main", {}).get("aqi")
        components = item.get("components", {})
        pm25 = components.get("pm2_5", 0.0)
        row = {
            "timestamp": timestamp,
            "aqi": float(aqi) if aqi is not None else None,
            "us_aqi": pm25_to_us_aqi(pm25),
            "co": components.get("co", 0.0),
            "no": components.get("no", 0.0),
            "no2": components.get("no2", 0.0),
            "o3": components.get("o3", 0.0),
            "so2": components.get("so2", 0.0),
            "pm2_5": pm25,
            "pm10": components.get("pm10", 0.0),
            "nh3": components.get("nh3", 0.0),
            "source": "openweather",
        }
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)
    print(f"Successfully saved {len(dataframe)} records to {output_path}!")


if __name__ == "__main__":
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your_openweather_api_key_here":
        print("Error: API Key is missing or invalid. Please check your .env file.")
    else:
        end_time = int(time.time())
        start_time = end_time - (2 * 365 * 24 * 60 * 60)
        api_data = fetch_historical_aqi(DEFAULT_LAT, DEFAULT_LON, start_time, end_time)
        if api_data:
            process_and_save_data(api_data, "data/raw/historical_aqi.csv")
