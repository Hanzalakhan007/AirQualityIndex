"""Helpers for keeping the MongoDB model registry from growing without bound."""
from __future__ import annotations

from typing import Any

from gridfs import GridFSBucket

from config.settings import MONGO_MODEL_BUCKET, MONGO_MODEL_REGISTRY_COLLECTION


def ensure_model_registry_indexes(database: Any) -> None:
    registry = database[MONGO_MODEL_REGISTRY_COLLECTION]
    registry.create_index([("registry_name", 1), ("version", -1)])


def prune_model_registry_versions(database: Any, registry_name: str, keep_versions: int) -> int:
    """Delete older model-registry versions and their GridFS payloads."""
    if keep_versions < 0:
        raise ValueError("keep_versions must be zero or greater.")

    registry = database[MONGO_MODEL_REGISTRY_COLLECTION]
    bucket = GridFSBucket(database, bucket_name=MONGO_MODEL_BUCKET)
    stale_documents = list(
        registry.find(
            {"registry_name": registry_name},
            sort=[("version", -1), ("created_at", -1)],
        ).skip(keep_versions)
    )

    deleted_documents = 0
    for document in stale_documents:
        file_id = document.get("artifact_file_id")
        if file_id is not None:
            try:
                bucket.delete(file_id)
            except Exception:
                pass
        deleted_documents += registry.delete_one({"_id": document["_id"]}).deleted_count

    return deleted_documents
