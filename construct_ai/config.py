"""Shared configuration and constants.

Kept intentionally small and explicit for hackathon readability.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

HISTORY_CSV_PATH = DATA_DIR / "history.csv"
RISK_MODEL_PATH = MODELS_DIR / "risk_model.joblib"

RISK_FEATURES = [
    "temperature",
    "humidity",
    "workforce_availability",
    "material_strength",
    "equipment_reliability",
]

SENSOR_SIGNALS = [
    "temperature",
    "humidity",
    "vibration",
    "structural_load",
]
