"""Generative Design Engine.

Goal: given high-level constraints, generate multiple plausible construction system options,
score them on cost/durability/time, and recommend the best.

This is a simplified, judge-friendly simulation (not a structural design tool).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd


WeatherLabel = Literal["Ideal", "Mild", "Challenging", "Severe"]


@dataclass(frozen=True)
class DesignInputs:
    """High-level knobs a PM/architect might provide."""

    budget_usd: float
    material_quality: float  # 0..10 (higher = better)
    labor_efficiency: float  # 0..10 (higher = better)
    weather_severity: float  # 0..10 (higher = worse conditions)


@dataclass(frozen=True)
class DesignOption:
    """A generated conceptual design option."""

    design_id: str
    system: str
    foundation: str
    primary_structure: str
    envelope: str
    prefabrication_level: str

    estimated_cost_usd: float
    estimated_duration_days: float
    durability_index: float  # 0..100

    score_cost: float
    score_time: float
    score_durability: float
    score_total: float


def weather_bucket(weather_severity: float) -> WeatherLabel:
    if weather_severity < 2.5:
        return "Ideal"
    if weather_severity < 5.0:
        return "Mild"
    if weather_severity < 7.5:
        return "Challenging"
    return "Severe"


def generate_design_options(inputs: DesignInputs, n_options: int = 8, seed: int = 42) -> list[DesignOption]:
    """Generate multiple conceptual options.

    Uses a small design catalog with param-driven variability.
    """

    rng = np.random.default_rng(seed)

    # A compact “catalog” of plausible construction systems.
    catalog = [
        {
            "system": "Reinforced Concrete (RC)",
            "foundation": "Strip + Pad Footings",
            "primary_structure": "Cast-in-place RC frame",
            "envelope": "Masonry infill + plaster",
            "prefabrication_level": "Low",
            "base_cost": 1.00,
            "base_time": 1.00,
            "base_durability": 78,
        },
        {
            "system": "Structural Steel", 
            "foundation": "Pile Foundation",
            "primary_structure": "Steel moment frame",
            "envelope": "Insulated metal panels",
            "prefabrication_level": "Medium",
            "base_cost": 1.12,
            "base_time": 0.85,
            "base_durability": 75,
        },
        {
            "system": "Precast Concrete",
            "foundation": "Raft Foundation",
            "primary_structure": "Precast beams + columns",
            "envelope": "Precast cladding panels",
            "prefabrication_level": "High",
            "base_cost": 1.08,
            "base_time": 0.78,
            "base_durability": 82,
        },
        {
            "system": "CLT Hybrid",
            "foundation": "Raft Foundation",
            "primary_structure": "CLT panels + steel cores",
            "envelope": "Rainscreen facade",
            "prefabrication_level": "High",
            "base_cost": 1.05,
            "base_time": 0.80,
            "base_durability": 70,
        },
    ]

    # Baseline project scale derived from budget.
    baseline_cost_target = float(inputs.budget_usd)
    baseline_duration = 180.0  # ~6 months baseline

    options: list[DesignOption] = []

    for i in range(n_options):
        item = catalog[i % len(catalog)].copy()

        # Small controlled randomization so multiple options exist within same system.
        variability = rng.normal(loc=1.0, scale=0.05)

        # Effects from inputs (simple but grounded in terminology).
        quality_factor = 1.0 + 0.03 * (inputs.material_quality - 5.0)  # better quality costs more
        labor_time_factor = 1.0 - 0.04 * (inputs.labor_efficiency - 5.0)  # more efficient -> faster
        weather_time_factor = 1.0 + 0.05 * (inputs.weather_severity / 10.0)  # worse weather -> slower

        # Prefabrication helps schedule but can cost more.
        prefab = item["prefabrication_level"]
        prefab_time = {"Low": 1.00, "Medium": 0.93, "High": 0.86}[prefab]
        prefab_cost = {"Low": 1.00, "Medium": 1.03, "High": 1.06}[prefab]

        estimated_cost = baseline_cost_target * item["base_cost"] * quality_factor * prefab_cost * variability

        estimated_duration = (
            baseline_duration
            * item["base_time"]
            * labor_time_factor
            * weather_time_factor
            * prefab_time
            * (1.0 + rng.normal(0.0, 0.03))
        )
        estimated_duration = float(max(60.0, estimated_duration))

        # Durability index: higher with quality, some systems slightly higher.
        durability = (
            item["base_durability"]
            + 2.5 * (inputs.material_quality - 5.0)
            - 1.5 * (inputs.weather_severity - 5.0) / 2.0
            + rng.normal(0.0, 2.0)
        )
        durability = float(np.clip(durability, 40.0, 95.0))

        # Scoring: normalize to a 0..100-ish scale.
        # Cost: best when <= budget, penalize overruns.
        cost_overrun_ratio = (estimated_cost - baseline_cost_target) / max(1.0, baseline_cost_target)
        score_cost = 100.0 - 220.0 * max(0.0, cost_overrun_ratio) - 20.0 * abs(min(0.0, cost_overrun_ratio))

        # Time: lower duration -> higher score.
        score_time = 100.0 - (estimated_duration - baseline_duration) * 0.35

        # Durability: already near 0..100.
        score_durability = durability

        # Total score weights: make the demo look “smart” but intuitive.
        # Under severe weather, durability matters slightly more.
        w_d = 0.36 + 0.08 * (inputs.weather_severity / 10.0)
        w_c = 0.34
        w_t = 1.0 - (w_d + w_c)

        score_total = w_c * score_cost + w_t * score_time + w_d * score_durability

        options.append(
            DesignOption(
                design_id=f"D-{i+1:02d}",
                system=item["system"],
                foundation=item["foundation"],
                primary_structure=item["primary_structure"],
                envelope=item["envelope"],
                prefabrication_level=item["prefabrication_level"],
                estimated_cost_usd=float(estimated_cost),
                estimated_duration_days=float(estimated_duration),
                durability_index=durability,
                score_cost=float(score_cost),
                score_time=float(score_time),
                score_durability=float(score_durability),
                score_total=float(score_total),
            )
        )

    # Sort descending by total score.
    options.sort(key=lambda o: o.score_total, reverse=True)
    return options


def best_design(options: list[DesignOption]) -> DesignOption:
    if not options:
        raise ValueError("No design options were generated")
    return options[0]


def options_to_frame(options: list[DesignOption]) -> pd.DataFrame:
    """Convert to a DataFrame for charts/tables."""

    df = pd.DataFrame([asdict(o) for o in options])
    df["estimated_cost_usd"] = df["estimated_cost_usd"].round(0)
    df["estimated_duration_days"] = df["estimated_duration_days"].round(1)
    df["durability_index"] = df["durability_index"].round(1)
    df["score_total"] = df["score_total"].round(1)
    return df
