import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests import RequestException
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import (
    DEFAULT_LAT,
    DEFAULT_LON,
    OPEN_METEO_AIR_QUALITY,
    OPENWEATHER_AIR_POLLUTION_CURRENT,
    OPENWEATHER_AIR_POLLUTION_HISTORY,
    OPENWEATHER_API_KEY,
    RAW_DIR,
    TIMEZONE,
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
    try:
        response = requests.get(OPENWEATHER_AIR_POLLUTION_HISTORY, params=params, timeout=90)
    except RequestException as exc:
        print(f"Error fetching data: {exc}")
        return None
    if response.status_code == 200:
        return response.json().get("list", [])
    print(f"Error fetching data: {response.status_code}")
    print(response.text)
    return None


def fetch_current_aqi(lat: float, lon: float) -> dict | None:
    """Fetch the latest current-hour pollutant measurement from OpenWeather."""
    print(f"Fetching current AQI for Lat: {lat}, Lon: {lon}...")
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
    }
    try:
        response = requests.get(OPENWEATHER_AIR_POLLUTION_CURRENT, params=params, timeout=30)
    except RequestException as exc:
        print(f"Error fetching current AQI: {exc}")
        return None
    if response.status_code == 200:
        rows = response.json().get("list", [])
        if rows:
            return rows[0]
        print("OpenWeather current AQI returned no rows.")
        return None
    print(f"Error fetching current AQI: {response.status_code}")
    print(response.text)
    return None


def fetch_openmeteo_historical_aqi(lat: float, lon: float, start_date: int, end_date: int) -> list[dict] | None:
    """Fetch Open-Meteo air-quality data in the same row shape used for feature building."""
    start_iso = datetime.fromtimestamp(start_date, tz=timezone.utc).date().isoformat()
    end_iso = datetime.fromtimestamp(end_date, tz=timezone.utc).date().isoformat()
    local_tz = ZoneInfo(TIMEZONE)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_iso,
        "end_date": end_iso,
        "hourly": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": TIMEZONE,
    }
    try:
        response = requests.get(OPEN_METEO_AIR_QUALITY, params=params, timeout=90)
        response.raise_for_status()
    except RequestException as exc:
        print(f"Error fetching Open-Meteo data: {exc}")
        return None

    hourly = response.json().get("hourly", {})
    timestamps = hourly.get("time", [])
    rows = []
    for index, timestamp in enumerate(timestamps):
        us_aqi = hourly.get("us_aqi", [None] * len(timestamps))[index]
        pm25 = hourly.get("pm2_5", [0.0] * len(timestamps))[index]
        rows.append(
            {
                "timestamp": datetime.fromisoformat(timestamp).replace(tzinfo=local_tz),
                "aqi": float(us_aqi) if us_aqi is not None else None,
                "us_aqi": float(us_aqi) if us_aqi is not None else pm25_to_us_aqi(pm25),
                "co": hourly.get("carbon_monoxide", [0.0] * len(timestamps))[index] or 0.0,
                "no": 0.0,
                "no2": hourly.get("nitrogen_dioxide", [0.0] * len(timestamps))[index] or 0.0,
                "o3": hourly.get("ozone", [0.0] * len(timestamps))[index] or 0.0,
                "so2": hourly.get("sulphur_dioxide", [0.0] * len(timestamps))[index] or 0.0,
                "pm2_5": pm25 or 0.0,
                "pm10": hourly.get("pm10", [0.0] * len(timestamps))[index] or 0.0,
                "nh3": 0.0,
                "source": "openmeteo",
            }
        )
    print(f"Fetched {len(rows)} rows from Open-Meteo for {start_iso} -> {end_iso}.")
    return rows


def process_and_save_data(raw_data: list[dict], output_path: str) -> None:
    """Process raw OpenWeather AQI data into a training-ready CSV."""
    if not raw_data:
        print("No data to process.")
        return

    rows = []
    local_tz = ZoneInfo(TIMEZONE)
    for item in raw_data:
        if item.get("source") == "openmeteo":
            rows.append(item)
            continue
        timestamp = datetime.fromtimestamp(item["dt"], tz=timezone.utc).astimezone(local_tz)
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


def process_raw_data(raw_data: list[dict]) -> pd.DataFrame:
    """Process raw OpenWeather AQI data into a dataframe."""
    if not raw_data:
        return pd.DataFrame()

    rows = []
    local_tz = ZoneInfo(TIMEZONE)
    for item in raw_data:
        if item.get("source") == "openmeteo":
            rows.append(item)
            continue
        timestamp = datetime.fromtimestamp(item["dt"], tz=timezone.utc).astimezone(local_tz)
        aqi = item.get("main", {}).get("aqi")
        components = item.get("components", {})
        pm25 = components.get("pm2_5", 0.0)
        rows.append(
            {
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
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your_openweather_api_key_here":
        print("Error: API Key is missing or invalid. Please check your .env file.")
    else:
        end_time = int(time.time())
        start_time = end_time - (2 * 365 * 24 * 60 * 60)
        raw_data = fetch_historical_aqi(DEFAULT_LAT, DEFAULT_LON, start_time, end_time)
        if not raw_data:
            print("OpenWeather returned no historical rows; trying Open-Meteo fallback.")
            raw_data = fetch_openmeteo_historical_aqi(DEFAULT_LAT, DEFAULT_LON, start_time, end_time)
        if raw_data:
            output_path = RAW_DIR / "historical_aqi.csv"
            process_and_save_data(raw_data, str(output_path))
            dataframe = process_raw_data(raw_data)
            print(f"Prepared {len(dataframe)} raw AQI rows.")
        else:
            print("No historical AQI data was fetched from either provider.")
