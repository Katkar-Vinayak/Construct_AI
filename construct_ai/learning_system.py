"""Learning System.

Stores past project runs in CSV and extracts simple insights to improve recommendations.

This is intentionally lightweight:
- CSV as a persistence layer
- Aggregations + heuristics to demonstrate "learning from experience"
- Optional retraining set for the delay model (if labeled outcomes exist)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .config import HISTORY_CSV_PATH, RISK_FEATURES


HISTORY_COLUMNS = [
    "timestamp",
    "project_name",
    "budget_usd",
    "material_quality",
    "labor_efficiency",
    "weather_severity",
    *RISK_FEATURES,
    "selected_design_id",
    "design_system",
    "estimated_cost_usd",
    "estimated_duration_days",
    "durability_index",
    "risk_probability",
    "ai_decision_summary",
    "actual_delay_days",
    "notes",
]


@dataclass(frozen=True)
class LearningInsights:
    """Human-readable insights derived from history."""

    n_records: int
    n_labeled: int
    common_delay_drivers: list[str]
    recommended_schedule_buffer_pct: float
    recommended_extra_curing_days: float


def ensure_history_exists() -> None:
    HISTORY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HISTORY_CSV_PATH.exists():
        return

    df = pd.DataFrame(columns=HISTORY_COLUMNS)
    df.to_csv(HISTORY_CSV_PATH, index=False)


def load_history() -> pd.DataFrame:
    ensure_history_exists()
    df = pd.read_csv(HISTORY_CSV_PATH)

    # Align to expected schema (helps if the CSV was created with an older header).
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[HISTORY_COLUMNS]


def append_history(record: dict) -> None:
    """Append a single row to history CSV (creates file if missing)."""

    ensure_history_exists()
    df = load_history()

    # Normalize timestamp.
    record = dict(record)
    record.setdefault("timestamp", datetime.utcnow().isoformat())

    # Ensure all columns exist.
    for col in HISTORY_COLUMNS:
        record.setdefault(col, None)

    df = pd.concat([df, pd.DataFrame([record])[HISTORY_COLUMNS]], ignore_index=True)
    df.to_csv(HISTORY_CSV_PATH, index=False)


def compute_learning_insights(history: pd.DataFrame) -> LearningInsights:
    if history.empty:
        return LearningInsights(
            n_records=0,
            n_labeled=0,
            common_delay_drivers=["No history yet"],
            recommended_schedule_buffer_pct=6.0,
            recommended_extra_curing_days=0.5,
        )

    labeled = history.dropna(subset=["actual_delay_days"])
    n_labeled = int(len(labeled))

    drivers: list[str] = []

    # Derive simple signals from labeled projects.
    if n_labeled >= 5:
        delayed = labeled[labeled["actual_delay_days"].astype(float) > 0]
        if len(delayed) >= 3:
            # Compare means to infer drivers.
            base = labeled.copy()
            for f in RISK_FEATURES:
                try:
                    if delayed[f].astype(float).mean() > base[f].astype(float).mean():
                        drivers.append(f"Higher {f.replace('_', ' ')} associated with delays")
                    else:
                        drivers.append(f"Lower {f.replace('_', ' ')} associated with delays")
                except Exception:
                    continue

    if not drivers:
        drivers = ["Limited labeled outcomes; using baseline heuristics"]

    # Recommendations: modest adjustments based on average delay.
    avg_delay = float(labeled["actual_delay_days"].astype(float).mean()) if n_labeled else 0.0

    schedule_buffer = 6.0 + min(8.0, max(0.0, avg_delay)) * 0.6

    # Curing suggestion based on humidity (if present).
    if "humidity" in history.columns and history["humidity"].notna().any():
        humid_mean = float(history["humidity"].dropna().astype(float).mean())
    else:
        humid_mean = 60.0

    extra_curing = 0.5 + max(0.0, (humid_mean - 65.0)) / 40.0  # up to ~1.4 days

    return LearningInsights(
        n_records=int(len(history)),
        n_labeled=n_labeled,
        common_delay_drivers=drivers[:5],
        recommended_schedule_buffer_pct=float(round(schedule_buffer, 1)),
        recommended_extra_curing_days=float(round(extra_curing, 1)),
    )
