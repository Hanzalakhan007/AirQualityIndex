"""Shared project settings and filesystem paths."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPLAINABILITY_DIR = REPORTS_DIR / "explainability"
STREAMLIT_DIR = PROJECT_ROOT / ".streamlit"

for directory in (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    EXPLAINABILITY_DIR,
    STREAMLIT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_CITY = os.getenv("AQI_CITY", os.getenv("CITY", "Karachi"))
DEFAULT_LAT = float(os.getenv("AQI_LAT", os.getenv("LAT", "24.8607")))
DEFAULT_LON = float(os.getenv("AQI_LON", os.getenv("LON", "67.0011")))
TIMEZONE = os.getenv("AQI_TIMEZONE", "Asia/Karachi")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")

FEATURE_GROUP_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP", "aqi_features")
FEATURE_GROUP_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))

USE_OPENMETEO_AQI = os.getenv("USE_OPENMETEO_AQI", "true").lower() in ("true", "1", "yes")
OPEN_METEO_BASE = "https://api.open-meteo.com/v1"
OPEN_METEO_AIR_QUALITY = "https://air-quality.api.open-meteo.com/v1/air-quality"
OPENWEATHER_AIR_POLLUTION_HISTORY = "https://api.openweathermap.org/data/2.5/air_pollution/history"

AQI_SCALE_1_5 = os.getenv("AQI_SCALE_1_5", "false").lower() in ("true", "1", "yes")
AQI_UNHEALTHY_THRESHOLD = 150
AQI_HAZARDOUS_THRESHOLD = 200
KARACHI_REFERENCE_AQI = float(os.getenv("KARACHI_REFERENCE_AQI", "96.0"))
AQI_CALIBRATION_ENABLED = os.getenv("AQI_CALIBRATION_ENABLED", "true").lower() in ("true", "1", "yes")

MODEL_REGISTRY_NAMES = {
    "Ridge Regression": ("aqi_ridge_model", "ridge_model.pkl", "joblib"),
    "Random Forest": ("aqi_rf_model", "rf_model.pkl", "joblib"),
    "XGBoost": ("aqi_xgboost_model", "xgb_model.pkl", "joblib"),
    "PyTorch Deep Learning": ("aqi_pytorch_model", "pytorch_model.pth", "pytorch"),
}

MODEL_REGISTRY_FALLBACK_VERSIONS = {
    "Ridge Regression": 1,
    "Random Forest": 1,
    "XGBoost": 2,
    "PyTorch Deep Learning": 1,
}

SCALER_REGISTRY_NAME = "aqi_scaler"
SCALER_REGISTRY_FILENAME = "scaler.pkl"
