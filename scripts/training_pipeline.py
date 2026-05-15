"""Run the model training pipeline."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "models" / "train_model.py")], check=True)
