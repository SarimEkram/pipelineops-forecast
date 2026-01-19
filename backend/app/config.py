# backend/app/config.py
import os
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MAX_UPLOAD_MB = _int_env("MAX_UPLOAD_MB", 50)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
