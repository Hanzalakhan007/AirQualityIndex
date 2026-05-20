"""Shared inference helpers for the dashboard and API."""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from gridfs import GridFSBucket
from pymongo import MongoClient
from zoneinfo import ZoneInfo

try:
    import certifi
except ImportError:  # pragma: no cover - optional dependency
    certifi = None

from config.settings import (
    AQI_CALIBRATION_ENABLED,
    AQI_HAZARDOUS_THRESHOLD,
    AQI_UNHEALTHY_THRESHOLD,
    DEFAULT_CITY,
    DEFAULT_LAT,
    DEFAULT_LON,
    EXPLAINABILITY_DIR,
    KARACHI_REFERENCE_AQI,
    MODEL_REGISTRY_NAMES,
    MODELS_DIR,
    MONGO_DB_NAME,
    MONGO_FEATURES_COLLECTION,
    MONGO_MODEL_BUCKET,
    MONGO_MODEL_REGISTRY_COLLECTION,
    MONGO_URI,
    OPEN_METEO_AIR_QUALITY,
    PROCESSED_DIR,
    REPORTS_DIR,
    SCALER_REGISTRY_FILENAME,
    SCALER_REGISTRY_NAME,
    TIMEZONE,
)
from src.schema import FEATURE_COLUMNS

MODEL_OPTIONS = ["Best Available"] + list(MODEL_REGISTRY_NAMES.keys())
SLIDER_FEATURES = ["co", "no2", "o3", "pm2_5", "pm10", "nh3"]
LOCAL_FEATURES_FILE = PROCESSED_DIR / "features.csv"
LOCAL_METRICS_FILE = REPORTS_DIR / "model_metrics.csv"
LOCAL_MODEL_FILES = {
    "Ridge Regression": MODELS_DIR / "ridge_model.pkl",
    "Random Forest": MODELS_DIR / "rf_model.pkl",
    "XGBoost": MODELS_DIR / "xgb_model.pkl",
}
LOCAL_SCALER_FILE = MODELS_DIR / SCALER_REGISTRY_FILENAME


def clear_caches() -> None:
    """Clear cached models and feature data."""
    load_feature_data.cache_clear()
    load_models.cache_clear()
    get_model_registry_metadata.cache_clear()
    get_mongo_database.cache_clear()


def normalize_aqi_value(value: float | None, pm25: float | None = None) -> float | None:
    """Prefer already-normalized AQI values and fall back to PM2.5 conversion or 1-5 mapping."""
    if value is not None and not pd.isna(value):
        value = float(value)
        if value > 10:
            return value
        return min(500.0, max(0.0, value * 50.0))
    if pm25 is None or pd.isna(pm25):
        return None
    pm25 = float(pm25)
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
        if low_cp <= pm25 <= high_cp:
            return round(((high_aqi - low_aqi) / (high_cp - low_cp)) * (pm25 - low_cp) + low_aqi, 1)
    return 500.0


def calibrate_aqi(stored_aqi: float | None, reference_aqi: float = KARACHI_REFERENCE_AQI) -> float | None:
    """Dampen obviously unrealistic AQI spikes while keeping the same AQI band."""
    if stored_aqi is None or not AQI_CALIBRATION_ENABLED:
        return stored_aqi
    stored_aqi = float(stored_aqi)
    if stored_aqi > reference_aqi * 1.3:
        excess = stored_aqi - reference_aqi
        return max(0.0, min(500.0, reference_aqi + (excess * 0.15)))
    if stored_aqi < reference_aqi * 0.5:
        return (stored_aqi + reference_aqi) / 2
    return stored_aqi


def aqi_level_and_color(aqi: float | None) -> tuple[str, str]:
    if aqi is None or pd.isna(aqi):
        return "N/A", "#94A3B8"
    value = float(aqi)
    if value <= 50:
        return "Good", "#00E400"
    if value <= 100:
        return "Moderate", "#FFFF00"
    if value <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"
    if value <= 200:
        return "Unhealthy", "#FF0000"
    return "Hazardous", "#7E0023"


def health_recommendation(aqi: float | None) -> str:
    if aqi is None or pd.isna(aqi):
        return "No current AQI reading is available."
    value = float(aqi)
    if value <= 50:
        return "Air quality is satisfactory and outdoor activity is generally safe."
    if value <= 100:
        return "Sensitive groups should reduce prolonged outdoor exertion."
    if value <= 150:
        return "Children, seniors, and sensitive groups should limit time outdoors."
    if value <= 200:
        return "Outdoor exposure should be reduced and a mask is recommended."
    return "Avoid prolonged outdoor activity and stay indoors when possible."


@lru_cache(maxsize=1)
def get_mongo_database():
    client_kwargs = {
        "serverSelectionTimeoutMS": 8000,
        "socketTimeoutMS": 20000,
        "connectTimeoutMS": 20000,
    }
    if certifi is not None:
        client_kwargs["tlsCAFile"] = certifi.where()
    try:
        client = MongoClient(MONGO_URI, **client_kwargs)
        client.admin.command("ping")
    except Exception as exc:
        raise RuntimeError(f"Unable to connect to MongoDB: {exc}") from exc
    return client[MONGO_DB_NAME]


def _latest_registry_document(registry_name: str) -> dict[str, Any] | None:
    database = get_mongo_database()
    return database[MONGO_MODEL_REGISTRY_COLLECTION].find_one(
        {"registry_name": registry_name},
        sort=[("version", -1), ("created_at", -1)],
    )


def _download_artifact_bytes(file_id: Any) -> BytesIO | None:
    if file_id is None:
        return None
    try:
        bucket = GridFSBucket(get_mongo_database(), bucket_name=MONGO_MODEL_BUCKET)
        stream = bucket.open_download_stream(file_id)
        return BytesIO(stream.read())
    except Exception:
        return None


def _load_local_feature_data() -> pd.DataFrame:
    if not LOCAL_FEATURES_FILE.exists():
        raise RuntimeError(
            "MongoDB is unavailable and no local fallback feature file was found at "
            f"'{LOCAL_FEATURES_FILE}'."
        )
    dataframe = pd.read_csv(LOCAL_FEATURES_FILE)
    if "timestamp" not in dataframe.columns:
        raise RuntimeError(f"Local fallback feature file '{LOCAL_FEATURES_FILE}' does not contain a timestamp column.")
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    return dataframe.sort_values("timestamp").reset_index(drop=True)


def _load_local_metrics_metadata() -> dict[str, dict[str, Any]]:
    if not LOCAL_METRICS_FILE.exists():
        return {}

    dataframe = pd.read_csv(LOCAL_METRICS_FILE)
    metadata: dict[str, dict[str, Any]] = {}
    for _, row in dataframe.iterrows():
        label = str(row.get("Model", "")).strip()
        if not label:
            continue
        metrics = {
            "rmse": float(row.get("RMSE", 999999.0)),
            "mae": float(row.get("MAE", 999999.0)),
            "r2": float(row.get("R2_Score", row.get("R2", -999999.0))),
            "selection_score": float(row.get("Selection_Score", row.get("RMSE", 999999.0))),
            "fit_status": str(row.get("Fit_Status", "unknown")),
            "train_rmse": float(row.get("Train_RMSE", row.get("RMSE", 999999.0))),
            "rmse_gap": float(row.get("RMSE_Gap", 0.0)),
        }
        metadata[label] = {
            "registry_name": MODEL_REGISTRY_NAMES.get(label, ("", "", ""))[0],
            "metrics": metrics,
            "filename": LOCAL_MODEL_FILES.get(label, Path("")).name,
            "feature_count": len(FEATURE_COLUMNS),
        }
    return metadata


def _load_local_models(requested_models: tuple[str, ...] | None = None) -> tuple[dict[str, Any], Any, dict[str, dict[str, Any]]]:
    selected_labels = requested_models or tuple(MODEL_REGISTRY_NAMES.keys())
    scaler = None
    if LOCAL_SCALER_FILE.exists():
        scaler = joblib.load(LOCAL_SCALER_FILE)

    models: dict[str, Any] = {}
    metadata = _load_local_metrics_metadata()
    for label in selected_labels:
        model_path = LOCAL_MODEL_FILES.get(label)
        if model_path is None or not model_path.exists():
            continue
        models[label] = joblib.load(model_path)
        metadata.setdefault(
            label,
            {
                "registry_name": MODEL_REGISTRY_NAMES.get(label, ("", "", ""))[0],
                "metrics": {},
                "filename": model_path.name,
                "feature_count": len(FEATURE_COLUMNS),
            },
        )

    if scaler is None:
        raise RuntimeError(
            "MongoDB is unavailable and no local fallback scaler file was found at "
            f"'{LOCAL_SCALER_FILE}'."
        )
    if not models:
        raise RuntimeError("MongoDB is unavailable and no local fallback model artifacts were found.")
    return models, scaler, metadata


@lru_cache(maxsize=1)
def get_available_model_names() -> list[str]:
    """Return only the models that can actually be loaded right now."""
    try:
        models, scaler, _ = load_models()
        if scaler is None:
            return []
        return [label for label in MODEL_REGISTRY_NAMES if label in models]
    except Exception:
        if not LOCAL_SCALER_FILE.exists():
            return []
        return [label for label in MODEL_REGISTRY_NAMES if LOCAL_MODEL_FILES.get(label, Path("")).exists()]


def get_available_model_options() -> list[str]:
    available_models = get_available_model_names()
    if not available_models:
        return MODEL_OPTIONS
    return ["Best Available"] + available_models


def _model_metrics_value(metrics: dict[str, Any], metric_name: str, default: float) -> float:
    try:
        return float(metrics.get(metric_name, default))
    except Exception:
        return default


def _align_input_frame(input_frame: pd.DataFrame, scaler: Any) -> pd.DataFrame:
    """Align prediction input to the feature order expected by the fitted scaler."""
    expected_columns = getattr(scaler, "feature_names_in_", None)
    if expected_columns is None:
        return input_frame

    aligned = input_frame.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            if column == "aqi" and "us_aqi" in aligned.columns:
                aligned[column] = aligned["us_aqi"]
            else:
                aligned[column] = 0.0
    return aligned.loc[:, list(expected_columns)]


@lru_cache(maxsize=1)
def get_model_registry_metadata() -> dict[str, dict[str, Any]]:
    """Fetch model-registry metadata without downloading model artifacts."""
    metadata: dict[str, dict[str, Any]] = {}
    try:
        for label, (registry_name, _, _) in MODEL_REGISTRY_NAMES.items():
            document = _latest_registry_document(registry_name)
            if document is None:
                continue
            metadata[label] = {
                "registry_name": registry_name,
                "metrics": document.get("metrics", {}) or {},
                "version": document.get("version"),
                "feature_count": document.get("feature_count"),
            }
    except Exception:
        metadata = {}

    if metadata:
        return metadata

    metadata = _load_local_metrics_metadata()
    return metadata


@lru_cache(maxsize=8)
def load_models(requested_models: tuple[str, ...] | None = None) -> tuple[dict[str, Any], Any, dict[str, dict[str, Any]]]:
    """Load requested trained models and scaler from the MongoDB model registry."""
    selected_labels = requested_models or tuple(MODEL_REGISTRY_NAMES.keys())
    try:
        models: dict[str, Any] = {}
        metadata = get_model_registry_metadata()
        scaler = None
        scaler_document = _latest_registry_document(SCALER_REGISTRY_NAME)
        if scaler_document is not None:
            scaler_file = _download_artifact_bytes(scaler_document.get("artifact_file_id"))
            if scaler_file is not None:
                try:
                    scaler_file.seek(0)
                    scaler = joblib.load(scaler_file)
                except Exception:
                    scaler = None

        for label in selected_labels:
            if label not in MODEL_REGISTRY_NAMES:
                continue

            registry_name, filename, _model_kind = MODEL_REGISTRY_NAMES[label]
            document = _latest_registry_document(registry_name)
            if document is None:
                continue

            artifact_file = _download_artifact_bytes(document.get("artifact_file_id"))
            if artifact_file is None:
                continue

            try:
                artifact_file.seek(0)
                model = joblib.load(artifact_file)
            except Exception:
                model = None

            if model is None:
                continue

            models[label] = model
            metadata[label] = {
                "registry_name": registry_name,
                "metrics": document.get("metrics", {}) or {},
                "version": document.get("version"),
                "filename": document.get("filename", filename),
                "feature_count": document.get("feature_count"),
            }

        if scaler is not None and models:
            return models, scaler, metadata
    except Exception:
        pass

    return _load_local_models(selected_labels)


def get_model_leaderboard() -> list[dict[str, Any]]:
    """Return model-registry metrics sorted by best validation performance."""
    metadata = get_model_registry_metadata()
    available_models = set(get_available_model_names())
    leaderboard = []

    for label, info in metadata.items():
        if available_models and label not in available_models:
            continue
        metrics = info.get("metrics", {}) or {}
        rmse = _model_metrics_value(metrics, "rmse", 999999.0)
        mae = _model_metrics_value(metrics, "mae", 999999.0)
        selection_score = _model_metrics_value(metrics, "selection_score", rmse)
        r2 = _model_metrics_value(
            metrics,
            "r2",
            _model_metrics_value(metrics, "r2_avg", -999999.0),
        )
        leaderboard.append(
            {
                "model": label,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "selection_score": selection_score,
                "fit_status": metrics.get("fit_status", "unknown"),
                "version": info.get("version"),
            }
        )
    return sorted(leaderboard, key=lambda item: (item["selection_score"], item["rmse"], -item["r2"]))


def get_default_model_name() -> str:
    leaderboard = get_model_leaderboard()
    if leaderboard:
        return str(leaderboard[0]["model"])
    models, _, _ = load_models()
    if models:
        return next(iter(models))
    raise RuntimeError("No trained models are available.")


@lru_cache(maxsize=1)
def load_feature_data() -> pd.DataFrame:
    """Load feature data from the MongoDB feature store."""
    try:
        collection = get_mongo_database()[MONGO_FEATURES_COLLECTION]
        rows = list(collection.find({}, {"_id": 0}).sort("timestamp", 1))
        if rows:
            dataframe = pd.DataFrame(rows)
            dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
            return dataframe.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        pass

    return _load_local_feature_data()


def get_latest_feature_row() -> pd.Series:
    dataframe = load_feature_data()
    if dataframe.empty:
        raise ValueError("Feature dataset is empty.")
    return dataframe.iloc[-1]


def build_input_frame(latest_row: pd.Series, overrides: dict[str, float] | None = None) -> pd.DataFrame:
    """Create a single-row feature frame aligned to the model training columns."""
    overrides = overrides or {}
    now_local = datetime.now(ZoneInfo(TIMEZONE))
    current_us_aqi = normalize_aqi_value(
        latest_row.get("us_aqi", latest_row.get("aqi")),
        latest_row.get("pm2_5"),
    ) or 0.0

    frame = {
        "aqi": float(overrides.get("aqi", current_us_aqi)),
        "co": float(overrides.get("co", latest_row.get("co", 0.0))),
        "no": float(overrides.get("no", latest_row.get("no", 0.0))),
        "no2": float(overrides.get("no2", latest_row.get("no2", 0.0))),
        "o3": float(overrides.get("o3", latest_row.get("o3", 0.0))),
        "so2": float(overrides.get("so2", latest_row.get("so2", 0.0))),
        "pm2_5": float(overrides.get("pm2_5", latest_row.get("pm2_5", 0.0))),
        "pm10": float(overrides.get("pm10", latest_row.get("pm10", 0.0))),
        "nh3": float(overrides.get("nh3", latest_row.get("nh3", 0.0))),
        "hour": int(overrides.get("hour", now_local.hour)),
        "day_of_week": int(overrides.get("day_of_week", now_local.weekday())),
        "month": int(overrides.get("month", now_local.month)),
        "is_weekend": int(overrides.get("is_weekend", 1 if now_local.weekday() >= 5 else 0)),
        "aqi_lag_1": float(overrides.get("aqi_lag_1", latest_row.get("aqi_lag_1", current_us_aqi))),
        "aqi_lag_24": float(overrides.get("aqi_lag_24", latest_row.get("aqi_lag_24", current_us_aqi))),
        "aqi_lag_48": float(overrides.get("aqi_lag_48", latest_row.get("aqi_lag_48", current_us_aqi))),
        "aqi_lag_72": float(overrides.get("aqi_lag_72", latest_row.get("aqi_lag_72", current_us_aqi))),
        "aqi_rolling_24": float(overrides.get("aqi_rolling_24", latest_row.get("aqi_rolling_24", current_us_aqi))),
    }
    return pd.DataFrame([frame], columns=FEATURE_COLUMNS)


def fetch_openmeteo_snapshot() -> dict[str, Any] | None:
    """Fetch a lightweight current AQI snapshot from Open-Meteo for display fallback."""
    params = {
        "latitude": DEFAULT_LAT,
        "longitude": DEFAULT_LON,
        "current": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide",
        "timezone": TIMEZONE,
    }
    try:
        response = requests.get(OPEN_METEO_AIR_QUALITY, params=params, timeout=20)
        response.raise_for_status()
        return response.json().get("current")
    except Exception:
        return None


def predict_next_days(model_name: str | None = None, overrides: dict[str, float] | None = None) -> dict[str, Any]:
    """Generate a three-day AQI forecast."""
    metadata = get_model_registry_metadata()
    selected_model = get_default_model_name() if not model_name or model_name == "Best Available" else model_name
    available_models = get_available_model_names()
    if available_models and selected_model not in available_models:
        raise RuntimeError(
            f"Model '{selected_model}' is not available in the current deployment. "
            f"Available models: {', '.join(available_models)}."
        )
    models, scaler, _ = load_models((selected_model,))
    if scaler is None:
        raise RuntimeError("Scaler could not be loaded from MongoDB model registry.")
    if selected_model not in models:
        raise RuntimeError(f"Model '{selected_model}' is not available.")

    latest_row = get_latest_feature_row()
    input_frame = build_input_frame(latest_row, overrides)
    aligned_input = _align_input_frame(input_frame, scaler)
    scaled_input = scaler.transform(aligned_input)
    model = models[selected_model]

    raw_prediction = np.ravel(model.predict(scaled_input))

    predictions = [calibrate_aqi(max(1.0, float(value))) or 0.0 for value in raw_prediction[:3]]
    now_local = datetime.now(ZoneInfo(TIMEZONE))
    forecast_dates = [(now_local + timedelta(days=index + 1)).strftime("%b %d") for index in range(3)]

    return {
        "city": DEFAULT_CITY,
        "model_name": selected_model,
        "latest_row": latest_row,
        "input_frame": aligned_input,
        "predictions": predictions,
        "forecast_dates": forecast_dates,
        "model_metrics": metadata.get(selected_model, {}).get("metrics", {}),
        "leaderboard": get_model_leaderboard(),
    }


def get_current_aqi() -> tuple[float | None, str]:
    """Return the best current AQI reading and its source label."""
    latest_row = get_latest_feature_row()
    current_value = normalize_aqi_value(
        latest_row.get("us_aqi", latest_row.get("aqi")),
        latest_row.get("pm2_5"),
    )
    if current_value is not None:
        return calibrate_aqi(current_value), "Current Observed"

    snapshot = fetch_openmeteo_snapshot()
    if snapshot:
        snapshot_value = normalize_aqi_value(snapshot.get("us_aqi"), snapshot.get("pm2_5"))
        if snapshot_value is not None:
            return calibrate_aqi(snapshot_value), "Current Snapshot"
    return None, "Unavailable"


def get_recent_daily_history(days: int = 14) -> pd.DataFrame:
    """Return recent daily AQI history derived from the feature store."""
    dataframe = load_feature_data().copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    dataframe["aqi_display"] = dataframe.apply(
        lambda row: normalize_aqi_value(row.get("aqi", row.get("us_aqi")), row.get("pm2_5")),
        axis=1,
    )
    history = (
        dataframe.set_index("timestamp")["aqi_display"]
        .resample("D")
        .mean()
        .tail(days)
        .reset_index()
        .rename(columns={"timestamp": "date", "aqi_display": "aqi_display"})
    )
    return history


def build_forecast_curve(predictions: list[float]) -> pd.DataFrame:
    """Create a simple hourly forecast curve for the next 72 hours."""
    start = datetime.now(ZoneInfo(TIMEZONE)).replace(minute=0, second=0, microsecond=0)
    rows = []
    for day_index, prediction in enumerate(predictions, start=1):
        day_start = start + timedelta(days=day_index)
        for hour in range(24):
            rows.append(
                {
                    "timestamp": day_start + timedelta(hours=hour),
                    "aqi_predicted": prediction,
                    "day_label": f"Day {day_index}",
                }
            )
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe["hour"] = dataframe["timestamp"].dt.strftime("%b %d %H:%M")
    return dataframe


def alert_days(predictions: list[float]) -> list[dict[str, Any]]:
    alerts = []
    for index, value in enumerate(predictions, start=1):
        if value >= AQI_UNHEALTHY_THRESHOLD:
            alerts.append(
                {
                    "day": index,
                    "aqi": value,
                    "level": "Hazardous" if value >= AQI_HAZARDOUS_THRESHOLD else "Unhealthy",
                }
            )
    return alerts


def explainability_images() -> dict[str, Path]:
    """Return paths to explainability images when present."""
    image_names = {
        "shap_summary_bar": EXPLAINABILITY_DIR / "shap_summary_bar.png",
        "shap_impact": EXPLAINABILITY_DIR / "shap_impact.png",
        "lime_single_prediction": EXPLAINABILITY_DIR / "lime_single_prediction.png",
    }
    return {name: path for name, path in image_names.items() if path.exists()}
