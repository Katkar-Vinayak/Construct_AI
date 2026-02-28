"""IoT Sensor Simulator.

Simulates real-time sensor streams typical of a construction site / early operation:
- temperature (°C)
- humidity (%RH)
- vibration (mm/s, simplified)
- structural_load (kN, simplified)

The simulator is intentionally lightweight for Streamlit demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass
class SensorReading:
    timestamp: datetime
    temperature: float
    humidity: float
    vibration: float
    structural_load: float


class SensorSimulator:
    """Random-walk simulator with occasional events (equipment vibration spikes, load changes)."""

    def __init__(
        self,
        *,
        temperature_c: float = 28.0,
        humidity_pct: float = 55.0,
        vibration_mms: float = 1.2,
        structural_load_kn: float = 220.0,
        weather_severity: float = 4.0,
        seed: int = 7,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.temperature_c = float(temperature_c)
        self.humidity_pct = float(humidity_pct)
        self.vibration_mms = float(vibration_mms)
        self.structural_load_kn = float(structural_load_kn)
        self.weather_severity = float(weather_severity)

    def step(self) -> SensorReading:
        """Advance simulation by one tick (~1 second in the UI)."""

        # Weather severity increases volatility.
        vol = 1.0 + 0.6 * (self.weather_severity / 10.0)

        self.temperature_c += float(self.rng.normal(0.0, 0.15 * vol))
        self.humidity_pct += float(self.rng.normal(0.0, 0.35 * vol))

        # Vibration: mostly stable, but equipment can create spikes.
        self.vibration_mms += float(self.rng.normal(0.0, 0.08 * vol))
        if self.rng.random() < 0.06:
            self.vibration_mms += float(self.rng.uniform(1.0, 4.5) * vol)

        # Structural load changes more slowly; occasional heavy lift / pour event.
        self.structural_load_kn += float(self.rng.normal(0.0, 1.5 * vol))
        if self.rng.random() < 0.03:
            self.structural_load_kn += float(self.rng.uniform(10.0, 35.0) * vol)

        # Physical constraints.
        self.temperature_c = float(np.clip(self.temperature_c, -5.0, 48.0))
        self.humidity_pct = float(np.clip(self.humidity_pct, 10.0, 100.0))
        self.vibration_mms = float(np.clip(self.vibration_mms, 0.0, 12.0))
        self.structural_load_kn = float(np.clip(self.structural_load_kn, 50.0, 600.0))

        return SensorReading(
            timestamp=datetime.now(timezone.utc),
            temperature=self.temperature_c,
            humidity=self.humidity_pct,
            vibration=self.vibration_mms,
            structural_load=self.structural_load_kn,
        )
