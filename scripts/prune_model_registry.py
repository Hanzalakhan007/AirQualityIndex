"""Prune older MongoDB model-registry versions to free Atlas storage."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    MODEL_REGISTRY_NAMES,
    MONGO_DB_NAME,
    MONGO_MODEL_REGISTRY_MAX_VERSIONS,
    SCALER_REGISTRY_NAME,
)
from src.model_registry import ensure_model_registry_indexes, prune_model_registry_versions  # noqa: E402
from src.mongo import (  # noqa: E402
    create_verified_mongo_client,
    format_mongo_space_quota_error,
    is_mongo_space_quota_error,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        type=int,
        default=MONGO_MODEL_REGISTRY_MAX_VERSIONS,
        help="How many versions to keep per registry name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.keep < 1:
        raise SystemExit("--keep must be at least 1.")

    client = create_verified_mongo_client()
    try:
        database = client[MONGO_DB_NAME]
        ensure_model_registry_indexes(database)

        total_deleted = 0
        registry_names = [registry_name for registry_name, _, _ in MODEL_REGISTRY_NAMES.values()]
        registry_names.append(SCALER_REGISTRY_NAME)
        for registry_name in registry_names:
            deleted = prune_model_registry_versions(database, registry_name, keep_versions=args.keep)
            total_deleted += deleted
            print(f"{registry_name}: deleted {deleted} old version(s); keeping latest {args.keep}.")

        print(f"Pruning complete. Total deleted registry documents: {total_deleted}.")
    except Exception as error:
        if is_mongo_space_quota_error(error):
            raise SystemExit(format_mongo_space_quota_error(error)) from error
        raise SystemExit(str(error)) from error
    finally:
        client.close()


if __name__ == "__main__":
    main()
