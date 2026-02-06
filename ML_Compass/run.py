#!/usr/bin/env python
"""Run ML Compass CLI. From repo root: python ML_Compass/run.py [oracle|explain|debate|capacity|run_all|serve] ..."""
import sys
from pathlib import Path

# Add repo root so that ml_toolbox and ML_Compass are both importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ML_Compass.app import main_cli

if __name__ == "__main__":
    main_cli()
