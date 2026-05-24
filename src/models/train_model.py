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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import (
    MODEL_REGISTRY_NAMES,
    PROCESSED_DIR,
    MONGO_DB_NAME,
    MONGO_FEATURES_COLLECTION,
    MONGO_MODEL_BUCKET,
    MONGO_MODEL_REGISTRY_MAX_VERSIONS,
    MONGO_MODEL_REGISTRY_COLLECTION,
    SCALER_REGISTRY_FILENAME,
    SCALER_REGISTRY_NAME,
)
from src.model_registry import ensure_model_registry_indexes, prune_model_registry_versions
from src.models.pytorch_model import (
    build_pytorch_model,
    predict_pytorch_model,
    save_pytorch_checkpoint,
    serialize_pytorch_checkpoint,
)
from src.mongo import create_verified_mongo_client
from src.schema import FEATURE_COLUMNS

load_dotenv()
torch.manual_seed(42)
np.random.seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("Starting Machine Learning Pipeline...")


def _load_local_training_data() -> pd.DataFrame | None:
    local_features = PROCESSED_DIR / "features.csv"
    if not local_features.exists():
        return None
    print(f"Loading training data from local fallback file '{local_features}'...")
    dataframe = pd.read_csv(local_features)
    return dataframe.sort_values("timestamp").reset_index(drop=True)


def _snapshot_signature(dataframe: pd.DataFrame) -> tuple[pd.Timestamp | None, int, int]:
    if dataframe.empty:
        return None, 0, 0
    latest_timestamp = None
    if "timestamp" in dataframe.columns:
        timestamps = pd.to_datetime(dataframe["timestamp"], errors="coerce")
        if not timestamps.dropna().empty:
            latest_timestamp = timestamps.max()
    feature_coverage = sum(1 for column in FEATURE_COLUMNS if column in dataframe.columns)
    return latest_timestamp, feature_coverage, len(dataframe)


def load_training_data() -> pd.DataFrame:
    """Load processed features from MongoDB, preferring fresher local fallback data when needed."""
    local_dataframe = _load_local_training_data()
    try:
        print("Connecting to MongoDB feature store...")
        client = create_verified_mongo_client()
        collection = client[MONGO_DB_NAME][MONGO_FEATURES_COLLECTION]
        print(f"Fetching features from collection '{MONGO_DB_NAME}.{MONGO_FEATURES_COLLECTION}'...")
        rows = list(collection.find({}, {"_id": 0}).sort("timestamp", 1))
        if rows:
            mongo_dataframe = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            if local_dataframe is not None:
                mongo_latest, mongo_feature_count, mongo_row_count = _snapshot_signature(mongo_dataframe)
                local_latest, local_feature_count, local_row_count = _snapshot_signature(local_dataframe)
                local_is_fresher = (
                    local_latest is not None
                    and (mongo_latest is None or local_latest > mongo_latest)
                )
                local_is_richer = local_feature_count > mongo_feature_count
                local_is_larger_same_horizon = (
                    local_latest is not None
                    and mongo_latest is not None
                    and local_latest == mongo_latest
                    and local_row_count > mongo_row_count
                )
                if local_is_fresher or local_is_richer or local_is_larger_same_horizon:
                    print(
                        "Local fallback feature file is newer or has richer engineered features than the MongoDB "
                        "snapshot. Using local training data for this run."
                    )
                    return local_dataframe
            return mongo_dataframe
    except Exception as exc:
        print(f"MongoDB training-data load failed, falling back to local CSV: {exc}")

    if local_dataframe is None:
        raise RuntimeError(
            "MongoDB feature collection is unavailable and no local fallback feature file was found. "
            "Run `python scripts/feature_pipeline.py` first."
        )
    return local_dataframe


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
    """Build calendar-day AQI targets for today plus the next three days."""
    result = dataframe.copy()
    bounded_target = pd.to_numeric(target, errors="coerce").clip(lower=0, upper=500)
    result["date"] = pd.to_datetime(result["timestamp"]).dt.floor("D")
    daily_targets = (
        pd.DataFrame({"date": result["date"], "aqi_target": bounded_target})
        .groupby("date", as_index=False)["aqi_target"]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )
    for day in range(4):
        daily_targets[f"target_day_{day}"] = daily_targets["aqi_target"].shift(-day)
    result = result.merge(
        daily_targets[["date", "target_day_0", "target_day_1", "target_day_2", "target_day_3"]],
        on="date",
        how="left",
    )
    return result


dataframe = load_training_data()
dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

aqi_target = get_target_series(dataframe)
dataframe = add_forecast_targets(dataframe, aqi_target)
dataframe = dataframe.dropna().reset_index(drop=True)

available_features = [column for column in FEATURE_COLUMNS if column in dataframe.columns]
X = dataframe[available_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
y = dataframe[["target_day_0", "target_day_1", "target_day_2", "target_day_3"]]

train_size = int(len(dataframe) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def train_pytorch_regressor(X_train_values: np.ndarray, y_train_values: np.ndarray):
    """Train a lightweight PyTorch regressor with a chronological validation split."""
    features = np.asarray(X_train_values, dtype=np.float32)
    targets = np.asarray(y_train_values, dtype=np.float32)
    split_index = max(1, int(len(features) * 0.85))
    if split_index >= len(features):
        split_index = max(1, len(features) - 1)

    train_features, val_features = features[:split_index], features[split_index:]
    train_targets, val_targets = targets[:split_index], targets[split_index:]
    if len(val_features) == 0:
        val_features, val_targets = train_features, train_targets

    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_targets),
    )
    batch_size = min(64, max(8, len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    model = build_pytorch_model(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_state = None
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    val_features_tensor = torch.from_numpy(np.asarray(val_features, dtype=np.float32))
    val_targets_tensor = torch.from_numpy(np.asarray(val_targets, dtype=np.float32))

    for _epoch in range(80):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_features_tensor)
            val_loss = float(loss_fn(val_predictions, val_targets_tensor).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


baseline_preds = np.tile(
    X_test["aqi_rolling_24"].fillna(X_test["aqi"]).to_numpy().reshape(-1, 1),
    (1, 4),
)

print("\n[1/4] Training Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_preds = ridge.predict(X_test_scaled)

print("[2/4] Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=8,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)

print("[3/4] Training XGBoost (Multi-Output)...")
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

print("[4/4] Training PyTorch MLP...")
pytorch_model = train_pytorch_regressor(X_train_scaled, y_train.to_numpy())
pytorch_preds = predict_pytorch_model(pytorch_model, X_test_scaled)

print("\n" + "=" * 40)
print("MODEL EVALUATION RESULTS (Test Set Average across Today + Next 3 Days)")
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
    "PyTorch MLP": evaluate("PyTorch MLP", y_test, pytorch_preds),
}

train_predictions = {
    "Ridge Regression": ridge.predict(X_train_scaled),
    "Random Forest": rf.predict(X_train_scaled),
    "XGBoost": xgb_model.predict(X_train_scaled),
    "PyTorch MLP": predict_pytorch_model(pytorch_model, X_train_scaled),
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

joblib.dump(ridge, "models/ridge_model.pkl")
joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")
save_pytorch_checkpoint("models/pytorch_model.pth", pytorch_model, X_train.shape[1], available_features)
joblib.dump(scaler, "models/scaler.pkl")
print("Saved refreshed local model artifacts in the 'models' folder.")

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
    database,
    registry_name: str,
    filename: str,
    model_kind: str,
    payload: bytes,
    metrics: dict[str, float | str | int],
    label: str | None = None,
) -> None:
    registry = database[MONGO_MODEL_REGISTRY_COLLECTION]
    bucket = GridFSBucket(database, bucket_name=MONGO_MODEL_BUCKET)
    ensure_model_registry_indexes(database)
    if MONGO_MODEL_REGISTRY_MAX_VERSIONS > 0:
        prune_model_registry_versions(
            database,
            registry_name,
            keep_versions=max(0, MONGO_MODEL_REGISTRY_MAX_VERSIONS - 1),
        )
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


mongo_client = None
try:
    mongo_client = create_verified_mongo_client()
    mongo_database = mongo_client[MONGO_DB_NAME]
    model_payloads = {
        "Ridge Regression": serialize_joblib_artifact(ridge),
        "Random Forest": serialize_joblib_artifact(rf),
        "XGBoost": serialize_joblib_artifact(xgb_model),
        "PyTorch MLP": serialize_pytorch_checkpoint(pytorch_model, X_train.shape[1], available_features),
    }
    for label, payload in model_payloads.items():
        registry_name, filename, model_kind = MODEL_REGISTRY_NAMES[label]
        save_registry_artifact(
            database=mongo_database,
            registry_name=registry_name,
            filename=filename,
            model_kind=model_kind,
            payload=payload,
            metrics=build_registry_metrics(label),
            label=label,
        )
    save_registry_artifact(
        database=mongo_database,
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
finally:
    if mongo_client is not None:
        mongo_client.close()
