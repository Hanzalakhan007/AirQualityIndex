"""Run the full hourly pipeline with a once-per-hour lease in MongoDB."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PIPELINE_KEY = "hourly_training_pipeline"

sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    MONGO_DB_NAME,
    MONGO_PIPELINE_CONTROL_COLLECTION,
)
from src.mongo import create_verified_mongo_client  # noqa: E402

load_dotenv()


def claim_hourly_slot(force_run: bool) -> tuple[bool, datetime]:
    slot_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if force_run:
        return True, slot_start

    client = create_verified_mongo_client()
    collection = client[MONGO_DB_NAME][MONGO_PIPELINE_CONTROL_COLLECTION]
    now = datetime.now(timezone.utc)
    collection.update_one(
        {"_id": PIPELINE_KEY},
        {"$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    claim = collection.find_one_and_update(
        {
            "_id": PIPELINE_KEY,
            "$or": [
                {"last_claimed_slot": {"$exists": False}},
                {"last_claimed_slot": {"$lt": slot_start}},
            ],
        },
        {
            "$set": {
                "last_claimed_slot": slot_start,
                "status": "running",
                "started_at": now,
                "updated_at": now,
            }
        },
    )
    return claim is not None, slot_start


def mark_pipeline_status(status: str, slot_start: datetime, error_message: str | None = None) -> None:
    client = create_verified_mongo_client()
    collection = client[MONGO_DB_NAME][MONGO_PIPELINE_CONTROL_COLLECTION]
    update = {
        "status": status,
        "updated_at": datetime.now(timezone.utc),
    }
    if status == "completed":
        update["last_completed_slot"] = slot_start
        update["last_success_at"] = datetime.now(timezone.utc)
    if error_message:
        update["last_error"] = error_message[:4000]
    collection.update_one({"_id": PIPELINE_KEY}, {"$set": update}, upsert=True)


def run_script(script_name: str) -> None:
    subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], check=True)


if __name__ == "__main__":
    force_run = os.getenv("PIPELINE_FORCE_RUN", "").lower() in {"1", "true", "yes"}
    claimed, slot_start = claim_hourly_slot(force_run=force_run)
    if not claimed:
        print(f"Hourly pipeline already claimed for slot {slot_start.isoformat()}; skipping.")
        raise SystemExit(0)

    try:
        for script in ("feature_pipeline.py", "training_pipeline.py"):
            run_script(script)
    except subprocess.CalledProcessError as exc:
        mark_pipeline_status("failed", slot_start, str(exc))
        raise
    else:
        mark_pipeline_status("completed", slot_start)
