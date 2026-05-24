"""Shared MongoDB client helpers."""
from __future__ import annotations

from typing import Any

from pymongo import MongoClient
from pymongo.errors import OperationFailure

try:
    import certifi
except ImportError:  # pragma: no cover - optional dependency
    certifi = None

from config.settings import (
    MONGO_CONNECT_TIMEOUT_MS,
    MONGO_ENABLED,
    MONGO_SERVER_SELECTION_TIMEOUT_MS,
    MONGO_SOCKET_TIMEOUT_MS,
    MONGO_URI,
)


def mongo_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "serverSelectionTimeoutMS": MONGO_SERVER_SELECTION_TIMEOUT_MS,
        "socketTimeoutMS": MONGO_SOCKET_TIMEOUT_MS,
        "connectTimeoutMS": MONGO_CONNECT_TIMEOUT_MS,
    }
    if certifi is not None:
        kwargs["tlsCAFile"] = certifi.where()
    return kwargs


def create_mongo_client() -> MongoClient:
    if not MONGO_ENABLED:
        raise RuntimeError("MongoDB is disabled by configuration.")
    return MongoClient(MONGO_URI, **mongo_client_kwargs())


def create_verified_mongo_client() -> MongoClient:
    client = create_mongo_client()
    client.admin.command("ping")
    return client


def is_mongo_space_quota_error(error: Exception) -> bool:
    """Return whether the exception represents an Atlas storage quota block."""
    message = str(error).lower()
    if "space quota" in message and "writes are blocked" in message:
        return True
    if isinstance(error, OperationFailure):
        details = getattr(error, "details", {}) or {}
        return details.get("code") == 8000 and "space quota" in message
    return False


def format_mongo_space_quota_error(error: Exception) -> str:
    """Build a clear remediation message for Atlas free-tier storage exhaustion."""
    return (
        "MongoDB Atlas is blocking writes because the cluster is over its storage quota. "
        "Free space by deleting old model-registry artifacts or scale the Atlas tier, then rerun the pipeline. "
        f"Original error: {error}"
    )
