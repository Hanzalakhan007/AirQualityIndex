"""Upload the current local model artifacts and metrics to the MongoDB model registry."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from gridfs import GridFSBucket

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    MODELS_DIR,
    MODEL_REGISTRY_NAMES,
    MONGO_DB_NAME,
    MONGO_MODEL_BUCKET,
    MONGO_MODEL_REGISTRY_COLLECTION,
    REPORTS_DIR,
    SCALER_REGISTRY_FILENAME,
    SCALER_REGISTRY_NAME,
)
from src.model_registry import ensure_model_registry_indexes, prune_model_registry_versions  # noqa: E402
from src.mongo import create_verified_mongo_client  # noqa: E402
from src.schema import FEATURE_COLUMNS  # noqa: E402

load_dotenv()

LOCAL_METRICS_FILE = REPORTS_DIR / "model_metrics.csv"


def read_local_metrics() -> dict[str, dict[str, float | str | int]]:
    if not LOCAL_METRICS_FILE.exists():
        raise SystemExit(f"Local metrics file not found: {LOCAL_METRICS_FILE}")

    dataframe = pd.read_csv(LOCAL_METRICS_FILE)
    metrics_by_label: dict[str, dict[str, float | str | int]] = {}
    for _, row in dataframe.iterrows():
        label = str(row.get("Model", "")).strip()
        if not label:
            continue
        metrics_by_label[label] = {
            "rmse": float(row.get("RMSE", 999999.0)),
            "mae": float(row.get("MAE", 999999.0)),
            "r2": float(row.get("R2_Score", row.get("R2", -999999.0))),
            "train_rmse": float(row.get("Train_RMSE", row.get("RMSE", 999999.0))),
            "rmse_gap": float(row.get("RMSE_Gap", 0.0)),
            "selection_score": float(row.get("Selection_Score", row.get("RMSE", 999999.0))),
            "fit_status": str(row.get("Fit_Status", "unknown")),
            "feature_count": len(FEATURE_COLUMNS),
        }
    return metrics_by_label


def upload_artifact(
    database,
    registry_name: str,
    filename: str,
    model_kind: str,
    payload: bytes,
    metrics: dict[str, float | str | int],
    label: str,
) -> None:
    registry = database[MONGO_MODEL_REGISTRY_COLLECTION]
    bucket = GridFSBucket(database, bucket_name=MONGO_MODEL_BUCKET)
    ensure_model_registry_indexes(database)
    prune_model_registry_versions(database, registry_name, keep_versions=2)

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
            "label": label,
            "registry_name": registry_name,
            "filename": filename,
            "model_kind": model_kind,
            "metrics": metrics,
            "version": version,
            "artifact_file_id": file_id,
            "feature_count": len(FEATURE_COLUMNS),
            "created_at": datetime.now(timezone.utc),
        }
    )


def main() -> None:
    metrics_by_label = read_local_metrics()
    client = create_verified_mongo_client()
    try:
        database = client[MONGO_DB_NAME]
        for label, (registry_name, filename, model_kind) in MODEL_REGISTRY_NAMES.items():
            model_path = MODELS_DIR / filename
            if not model_path.exists():
                raise SystemExit(f"Missing local model artifact for {label}: {model_path}")
            if label not in metrics_by_label:
                raise SystemExit(f"Missing metrics row for {label} in {LOCAL_METRICS_FILE}")
            upload_artifact(
                database=database,
                registry_name=registry_name,
                filename=filename,
                model_kind=model_kind,
                payload=model_path.read_bytes(),
                metrics=metrics_by_label[label],
                label=label,
            )
            print(f"Uploaded {label} as latest registry version.")

        scaler_path = MODELS_DIR / SCALER_REGISTRY_FILENAME
        if not scaler_path.exists():
            raise SystemExit(f"Missing local scaler artifact: {scaler_path}")
        upload_artifact(
            database=database,
            registry_name=SCALER_REGISTRY_NAME,
            filename=SCALER_REGISTRY_FILENAME,
            model_kind="joblib",
            payload=scaler_path.read_bytes(),
            metrics={"feature_count": len(FEATURE_COLUMNS)},
            label="Scaler",
        )
        print("Uploaded scaler as latest registry version.")
        print("Local healthy baseline is now synced to MongoDB model registry.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
