"""Run the full local AQI pipeline in sequence."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(script_name: str) -> None:
    subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], check=True)


if __name__ == "__main__":
    for script in ("fetch_raw_data.py", "feature_pipeline.py", "training_pipeline.py"):
        run_script(script)
