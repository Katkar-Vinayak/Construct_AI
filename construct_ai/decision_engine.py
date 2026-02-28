"""Adaptive Decision Engine.

When risks are detected (from ML prediction + sensor context), generate clear actions:
- adjust schedule
- recommend reinforcement
- suggest curing time
- optimize resource allocation

This is a rule-based layer on top of the ML output.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    severity: str  # Low/Medium/High
    title: str
    rationale: str
    recommended_action: str


def adaptive_decisions(
    *,
    risk_probability: float,
    temperature_c: float,
    humidity_pct: float,
    vibration_mms: float,
    structural_load_kn: float,
    workforce_availability: float,
) -> list[Decision]:
    decisions: list[Decision] = []

    if risk_probability >= 0.70:
        decisions.append(
            Decision(
                severity="High",
                title="Trigger schedule buffer + daily re-forecast",
                rationale=f"Delay risk is high ({risk_probability:.0%}).",
                recommended_action=(
                    "Add 10–15% float to critical path activities; run daily look-ahead planning and"
                    " lock procurement lead times."
                ),
            )
        )
    elif risk_probability >= 0.45:
        decisions.append(
            Decision(
                severity="Medium",
                title="Increase schedule contingency",
                rationale=f"Delay risk is moderate ({risk_probability:.0%}).",
                recommended_action="Add 5–8% buffer to weather-sensitive tasks; increase inspection cadence.",
            )
        )

    # Curing / concrete placement guidance.
    if humidity_pct >= 80.0 and temperature_c <= 18.0:
        decisions.append(
            Decision(
                severity="Medium",
                title="Extend curing time",
                rationale="High humidity + low temperature slows hydration and strength gain.",
                recommended_action="Increase curing duration by 1–2 days and verify early-age strength before stripping.",
            )
        )
    elif temperature_c >= 35.0:
        decisions.append(
            Decision(
                severity="Medium",
                title="Hot-weather concreting controls",
                rationale="High temperature accelerates set and increases cracking risk.",
                recommended_action="Use chilled water/ice, consider retarder, and implement aggressive curing + evaporation control.",
            )
        )

    # Structural health signals.
    if structural_load_kn >= 420.0 or vibration_mms >= 6.0:
        decisions.append(
            Decision(
                severity="High" if (structural_load_kn >= 480.0 or vibration_mms >= 8.0) else "Medium",
                title="Reinforcement / temporary works review",
                rationale="Elevated load/vibration can indicate staging issues or equipment-induced fatigue.",
                recommended_action=(
                    "Review shoring/reshoring plan, check connection tightness, and consider temporary bracing."
                    " Increase monitoring frequency."
                ),
            )
        )

    # Resource optimization.
    if workforce_availability <= 0.55:
        decisions.append(
            Decision(
                severity="Medium",
                title="Optimize crew allocation",
                rationale="Limited workforce availability increases cycle times and rework risk.",
                recommended_action="Re-sequence workfaces, prioritize critical trades, and pre-stage materials to reduce idle time.",
            )
        )

    if not decisions:
        decisions.append(
            Decision(
                severity="Low",
                title="Proceed with baseline plan",
                rationale="No significant risk signals detected.",
                recommended_action="Continue monitoring; keep weekly risk review active.",
            )
        )

    return decisions
