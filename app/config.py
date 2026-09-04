from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/productivity.db")
APP_ENV = os.getenv("APP_ENV", "development")

__all__ = ["BASE_DIR", "DATA_DIR", "DATABASE_URL", "APP_ENV"]
