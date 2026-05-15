"""Run exploratory data analysis."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "visualization" / "eda.py")], check=True)
