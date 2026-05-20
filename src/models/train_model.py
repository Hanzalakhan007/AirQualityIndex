import os
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from gridfs import GridFSBucket
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import (
    MODEL_REGISTRY_NAMES,
    MONGO_DB_NAME,
    MONGO_FEATURES_COLLECTION,
    MONGO_MODEL_BUCKET,
    MONGO_MODEL_REGISTRY_COLLECTION,
    MONGO_URI,
    SCALER_REGISTRY_FILENAME,
    SCALER_REGISTRY_NAME,
)
from src.schema import FEATURE_COLUMNS

load_dotenv()

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("Starting Machine Learning Pipeline...")

def load_training_data() -> pd.DataFrame:
    """Load processed features from the MongoDB feature store."""
    print("Connecting to MongoDB feature store...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    collection = client[MONGO_DB_NAME][MONGO_FEATURES_COLLECTION]
    print(f"Fetching features from collection '{MONGO_DB_NAME}.{MONGO_FEATURES_COLLECTION}'...")
    rows = list(collection.find({}, {"_id": 0}).sort("timestamp", 1))
    if not rows:
        raise RuntimeError("MongoDB feature collection is empty. Run `python scripts/feature_pipeline.py` first.")
    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)
    return dataframe


def get_target_series(dataframe: pd.DataFrame) -> pd.Series:
    """Prefer normalized AQI and only scale legacy OpenWeather 1-5 values."""
    if "us_aqi" in dataframe.columns:
        series = pd.to_numeric(dataframe["us_aqi"], errors="coerce")
        if not series.dropna().empty:
            return series
    base = pd.to_numeric(dataframe["aqi"], errors="coerce")
    if base.dropna().quantile(0.95) <= 10:
        return base * 50.0
    return base


def add_forecast_targets(dataframe: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Build realistic next-day average AQI targets on the same 0-500 scale."""
    result = dataframe.copy()
    bounded_target = pd.to_numeric(target, errors="coerce").clip(lower=0, upper=500)
    for day in range(1, 4):
        start_offset = (day - 1) * 24 + 1
        end_offset = day * 24 + 1
        shifted_hours = [bounded_target.shift(-offset) for offset in range(start_offset, end_offset)]
        result[f"target_day_{day}"] = pd.concat(shifted_hours, axis=1).mean(axis=1)
    return result


dataframe = load_training_data()
dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

aqi_target = get_target_series(dataframe)
dataframe = add_forecast_targets(dataframe, aqi_target)
dataframe = dataframe.dropna().reset_index(drop=True)

available_features = [column for column in FEATURE_COLUMNS if column in dataframe.columns]
X = dataframe[available_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
y = dataframe[["target_day_1", "target_day_2", "target_day_3"]]

train_size = int(len(dataframe) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_preds = np.tile(
    X_test["aqi_rolling_24"].fillna(X_test["aqi"]).to_numpy().reshape(-1, 1),
    (1, 3),
)

print("\n[1/3] Training Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_preds = ridge.predict(X_test_scaled)

print("[2/3] Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=8,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)

print("[3/3] Training XGBoost (Multi-Output)...")
xgb_base = xgb.XGBRegressor(
    n_estimators=250,
    max_depth=3,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
)
xgb_model = MultiOutputRegressor(xgb_base)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)

print("\n" + "=" * 40)
print("MODEL EVALUATION RESULTS (Test Set Average across 3 Days)")
print("=" * 40)


def score_predictions(y_true, y_pred) -> dict[str, float]:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, 500)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "prediction_min": float(np.nanmin(y_pred)),
        "prediction_max": float(np.nanmax(y_pred)),
    }


def evaluate(name, y_true, y_pred):
    metrics = score_predictions(y_true, y_pred)
    rmse = metrics["rmse"]
    mae = metrics["mae"]
    r2 = metrics["r2"]
    print(f"{name} prediction range: {np.nanmin(y_pred):.1f} to {np.nanmax(y_pred):.1f}")
    print(f"{name}:\n  Avg RMSE: {rmse:.4f} | Avg MAE: {mae:.4f} | Avg R2: {r2:.4f}\n")
    return metrics


baseline_results = {
    "Persistence Baseline": evaluate("Persistence Baseline", y_test, baseline_preds),
}
results = {
    "Ridge Regression": evaluate("Ridge Regression", y_test, ridge_preds),
    "Random Forest": evaluate("Random Forest", y_test, rf_preds),
    "XGBoost": evaluate("XGBoost", y_test, xgb_preds),
}

train_predictions = {
    "Ridge Regression": ridge.predict(X_train_scaled),
    "Random Forest": rf.predict(X_train_scaled),
    "XGBoost": xgb_model.predict(X_train_scaled),
}
for model_name, predictions in train_predictions.items():
    train_metrics = score_predictions(y_train, predictions)
    results[model_name]["train_rmse"] = train_metrics["rmse"]
    results[model_name]["train_mae"] = train_metrics["mae"]
    results[model_name]["train_r2"] = train_metrics["r2"]
    results[model_name]["rmse_gap"] = results[model_name]["rmse"] - train_metrics["rmse"]
    results[model_name]["overfit_ratio"] = results[model_name]["rmse"] / max(train_metrics["rmse"], 1e-9)
    results[model_name]["selection_score"] = results[model_name]["rmse"] + max(0.0, results[model_name]["rmse_gap"]) * 0.25
    results[model_name]["fit_status"] = (
        "overfit-risk"
        if results[model_name]["overfit_ratio"] > 1.8
        else "underfit-risk"
        if results[model_name]["r2"] < 0
        else "ok"
    )

best_model_name = min(results, key=lambda key: results[key]["selection_score"])
print(f"BEST MODEL: {best_model_name}")

pd.DataFrame(
    [
        {
            "Model": model_name,
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "R2_Score": metrics["r2"],
            "Train_RMSE": metrics["train_rmse"],
            "RMSE_Gap": metrics["rmse_gap"],
            "Fit_Status": metrics["fit_status"],
            "Selection_Score": metrics["selection_score"],
            "Is_Best_Model": model_name == best_model_name,
        }
        for model_name, metrics in results.items()
    ]
).to_csv("reports/model_metrics.csv", index=False)

print("\n" + "=" * 40)
print("UPLOADING TO MONGODB MODEL REGISTRY")
print("=" * 40)


def build_registry_metrics(model_name: str) -> dict[str, float | str | int]:
    metrics = dict(results[model_name])
    metrics["feature_count"] = len(available_features)
    return metrics


def serialize_joblib_artifact(artifact) -> bytes:
    buffer = BytesIO()
    joblib.dump(artifact, buffer)
    return buffer.getvalue()


def save_registry_artifact(
    registry_name: str,
    filename: str,
    model_kind: str,
    payload: bytes,
    metrics: dict[str, float | str | int],
    label: str | None = None,
) -> None:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    database = client[MONGO_DB_NAME]
    registry = database[MONGO_MODEL_REGISTRY_COLLECTION]
    bucket = GridFSBucket(database, bucket_name=MONGO_MODEL_BUCKET)
    latest = registry.find_one({"registry_name": registry_name}, sort=[("version", -1)])
    version = int((latest or {}).get("version", 0)) + 1
    file_id = bucket.upload_from_stream(
        filename,
        payload,
        metadata={
            "registry_name": registry_name,
            "version": version,
            "model_kind": model_kind,
        },
    )
    registry.insert_one(
        {
            "label": label or registry_name,
            "registry_name": registry_name,
            "filename": filename,
            "model_kind": model_kind,
            "metrics": metrics,
            "version": version,
            "artifact_file_id": file_id,
            "feature_count": len(available_features),
            "created_at": datetime.now(timezone.utc),
        }
    )
    registry.create_index([("registry_name", 1), ("version", -1)])


try:
    model_payloads = {
        "Ridge Regression": serialize_joblib_artifact(ridge),
        "Random Forest": serialize_joblib_artifact(rf),
        "XGBoost": serialize_joblib_artifact(xgb_model),
    }
    for label, payload in model_payloads.items():
        registry_name, filename, model_kind = MODEL_REGISTRY_NAMES[label]
        save_registry_artifact(
            registry_name=registry_name,
            filename=filename,
            model_kind=model_kind,
            payload=payload,
            metrics=build_registry_metrics(label),
            label=label,
        )
    save_registry_artifact(
        registry_name=SCALER_REGISTRY_NAME,
        filename=SCALER_REGISTRY_FILENAME,
        model_kind="joblib",
        payload=serialize_joblib_artifact(scaler),
        metrics={"feature_count": len(available_features)},
        label="Feature Scaler",
    )
    print("Successfully uploaded all models and scaler to MongoDB Registry!")
except Exception as exc:
    print(f"Failed to upload models to MongoDB: {exc}")
