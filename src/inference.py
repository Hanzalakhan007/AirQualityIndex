"""Shared inference helpers for the dashboard and API."""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from zoneinfo import ZoneInfo

from config.settings import (
    AQI_CALIBRATION_ENABLED,
    AQI_HAZARDOUS_THRESHOLD,
    AQI_UNHEALTHY_THRESHOLD,
    DEFAULT_CITY,
    DEFAULT_LAT,
    DEFAULT_LON,
    EXPLAINABILITY_DIR,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    KARACHI_REFERENCE_AQI,
    MODEL_REGISTRY_FALLBACK_VERSIONS,
    MODEL_REGISTRY_NAMES,
    MODELS_DIR,
    OPEN_METEO_AIR_QUALITY,
    PROCESSED_DIR,
    SCALER_REGISTRY_FILENAME,
    SCALER_REGISTRY_NAME,
    TIMEZONE,
)

FEATURE_COLUMNS = [
    "us_aqi",
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
    "pm25_rolling_24",
    "pm10_rolling_24",
    "co_rolling_24",
    "aqi_change_rate",
]

MODEL_OPTIONS = ["Best Available"] + list(MODEL_REGISTRY_NAMES.keys())
SLIDER_FEATURES = ["co", "no2", "o3", "pm2_5", "pm10", "nh3"]


class AQIPredictorNN(nn.Module):
    """Neural network used by the training pipeline."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.relu(self.fc1(inputs))
        outputs = self.relu(self.fc2(outputs))
        return self.fc3(outputs)


def clear_caches() -> None:
    """Clear cached models and feature data."""
    load_feature_data.cache_clear()
    load_models.cache_clear()


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


def _load_optional_hopsworks():
    try:
        import hopsworks  # type: ignore
    except Exception:
        return None
    try:
        return hopsworks.login()
    except Exception:
        return None


def _get_registry_model(project: Any, registry_name: str, fallback_version: int | None = None):
    try:
        model_registry = project.get_model_registry()
        try:
            registry_models = model_registry.get_models(name=registry_name)
            if registry_models:
                return max(registry_models, key=lambda item: int(getattr(item, "version", 0) or 0))
        except Exception:
            if fallback_version is not None:
                return model_registry.get_model(name=registry_name, version=fallback_version)
            return model_registry.get_model(name=registry_name)
    except Exception:
        return None


def _download_registry_file(model_obj: Any, expected_filename: str) -> Path | None:
    try:
        download_dir = Path(model_obj.download())
        exact_match = download_dir / expected_filename
        if exact_match.exists():
            return exact_match
        for candidate in download_dir.rglob(expected_filename):
            if candidate.is_file():
                return candidate
    except Exception:
        return None
    return None


def _load_pytorch_model(model_path: Path, feature_count: int | None = None) -> AQIPredictorNN | None:
    if not model_path.exists():
        return None
    try:
        input_dim_path = MODELS_DIR / "nn_input_dim.pkl"
        if input_dim_path.exists():
            input_dim = int(joblib.load(input_dim_path))
        elif feature_count is not None:
            input_dim = int(feature_count)
        else:
            input_dim = len(FEATURE_COLUMNS)
        model = AQIPredictorNN(input_dim)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception:
        return None


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
def load_models() -> tuple[dict[str, Any], Any, dict[str, dict[str, Any]]]:
    """Load latest trained models and scaler from Hopsworks or local disk."""
    models: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    scaler = None
    project = _load_optional_hopsworks()

    if project is not None:
        scaler_obj = _get_registry_model(project, SCALER_REGISTRY_NAME)
        if scaler_obj is not None:
            scaler_path = _download_registry_file(scaler_obj, SCALER_REGISTRY_FILENAME)
            if scaler_path is not None:
                try:
                    scaler = joblib.load(scaler_path)
                except Exception:
                    scaler = None

    if scaler is None:
        local_scaler = MODELS_DIR / SCALER_REGISTRY_FILENAME
        if local_scaler.exists():
            scaler = joblib.load(local_scaler)

    for label, (registry_name, filename, model_kind) in MODEL_REGISTRY_NAMES.items():
        model_obj = None
        metrics: dict[str, Any] = {}
        version = None

        if project is not None:
            model_obj = _get_registry_model(project, registry_name, MODEL_REGISTRY_FALLBACK_VERSIONS.get(label))
            if model_obj is not None:
                metrics = getattr(model_obj, "metrics", {}) or {}
                version = getattr(model_obj, "version", None)

        candidate_path: Path | None = _download_registry_file(model_obj, filename) if model_obj is not None else None
        if candidate_path is None:
            local_path = MODELS_DIR / filename
            if local_path.exists():
                candidate_path = local_path

        if candidate_path is None:
            continue

        if model_kind == "pytorch":
            model = _load_pytorch_model(candidate_path, metrics.get("feature_count"))
        else:
            try:
                model = joblib.load(candidate_path)
            except Exception:
                model = None

        if model is None:
            continue

        models[label] = model
        metadata[label] = {
            "registry_name": registry_name,
            "metrics": metrics,
            "version": version,
        }

    return models, scaler, metadata


def get_model_leaderboard() -> list[dict[str, Any]]:
    """Return model metrics sorted by best RMSE / R2."""
    _, _, metadata = load_models()
    leaderboard = []
    for label, info in metadata.items():
        metrics = info.get("metrics", {}) or {}
        leaderboard.append(
            {
                "model": label,
                "rmse": _model_metrics_value(metrics, "rmse", 999999.0),
                "mae": _model_metrics_value(metrics, "mae", 999999.0),
                "r2": _model_metrics_value(metrics, "r2", _model_metrics_value(metrics, "r2_avg", -999999.0)),
                "version": info.get("version"),
            }
        )
    if not leaderboard:
        metrics_path = Path("reports/model_metrics.csv")
        if metrics_path.exists():
            fallback = pd.read_csv(metrics_path)
            for _, row in fallback.iterrows():
                leaderboard.append(
                    {
                        "model": row.get("Model"),
                        "rmse": float(row.get("RMSE", 999999.0)),
                        "mae": float(row.get("MAE", 999999.0)),
                        "r2": float(row.get("R2_Score", -999999.0)),
                        "version": None,
                    }
                )
    return sorted(leaderboard, key=lambda item: (item["rmse"], -item["r2"]))


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
    """Load feature data from Hopsworks with a local CSV fallback."""
    project = _load_optional_hopsworks()
    if project is not None:
        try:
            feature_store = project.get_feature_store()
            feature_group = feature_store.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
            dataframe = feature_group.read()
            dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
            return dataframe.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            pass

    local_features = PROCESSED_DIR / "features.csv"
    if not local_features.exists():
        raise FileNotFoundError("No processed feature dataset found.")

    dataframe = pd.read_csv(local_features)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    return dataframe.sort_values("timestamp").reset_index(drop=True)


def get_latest_feature_row() -> pd.Series:
    dataframe = load_feature_data()
    if dataframe.empty:
        raise ValueError("Feature dataset is empty.")
    return dataframe.iloc[-1]


def build_input_frame(latest_row: pd.Series, overrides: dict[str, float] | None = None) -> pd.DataFrame:
    """Create a single-row feature frame aligned to the training columns."""
    overrides = overrides or {}
    now_local = datetime.now(ZoneInfo(TIMEZONE))
    current_us_aqi = normalize_aqi_value(
        latest_row.get("us_aqi", latest_row.get("aqi")),
        latest_row.get("pm2_5"),
    ) or 0.0

    frame = {
        "us_aqi": float(overrides.get("us_aqi", current_us_aqi)),
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
        "aqi_rolling_24": float(overrides.get("aqi_rolling_24", latest_row.get("aqi_rolling_24", current_us_aqi))),
        "pm25_rolling_24": float(overrides.get("pm25_rolling_24", latest_row.get("pm25_rolling_24", latest_row.get("pm2_5", 0.0)))),
        "pm10_rolling_24": float(overrides.get("pm10_rolling_24", latest_row.get("pm10_rolling_24", latest_row.get("pm10", 0.0)))),
        "co_rolling_24": float(overrides.get("co_rolling_24", latest_row.get("co_rolling_24", latest_row.get("co", 0.0)))),
        "aqi_change_rate": float(overrides.get("aqi_change_rate", latest_row.get("aqi_change_rate", 0.0))),
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
    models, scaler, metadata = load_models()
    if scaler is None:
        raise RuntimeError("Scaler could not be loaded from Hopsworks or local models.")

    selected_model = get_default_model_name() if not model_name or model_name == "Best Available" else model_name
    if selected_model not in models:
        raise RuntimeError(f"Model '{selected_model}' is not available.")

    latest_row = get_latest_feature_row()
    input_frame = build_input_frame(latest_row, overrides)
    aligned_input = _align_input_frame(input_frame, scaler)
    scaled_input = scaler.transform(aligned_input)
    model = models[selected_model]

    if selected_model == "PyTorch Deep Learning":
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
        with torch.no_grad():
            raw_prediction = model(tensor_input).cpu().numpy()[0]
    else:
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
    """Return recent daily AQI history for charting."""
    daily_path = PROCESSED_DIR / "daily_features.csv"
    if daily_path.exists():
        dataframe = pd.read_csv(daily_path)
        date_column = "date" if "date" in dataframe.columns else "timestamp"
        dataframe[date_column] = pd.to_datetime(dataframe[date_column])
        aqi_column = "us_aqi" if "us_aqi" in dataframe.columns else "aqi"
        dataframe["aqi_display"] = dataframe[aqi_column].apply(normalize_aqi_value)
        return dataframe.sort_values(date_column).tail(days).reset_index(drop=True)

    dataframe = load_feature_data().copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    dataframe["aqi_display"] = dataframe.apply(
        lambda row: normalize_aqi_value(row.get("us_aqi"), row.get("pm2_5")),
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
