"""Risk Prediction Model (scikit-learn).

Predicts probability of construction delay based on environmental + operational features.

Features (as required):
- temperature
- humidity
- workforce availability
- material strength
- equipment reliability

Model: Logistic Regression (simple, explainable).
Explainability: coefficient-based local contribution (logit decomposition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RISK_FEATURES, RISK_MODEL_PATH


@dataclass(frozen=True)
class RiskPrediction:
    probability_delay: float
    label: str


class RiskPredictor:
    """Wraps a scikit-learn pipeline and provides train/predict/explain helpers."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    @staticmethod
    def build_default_pipeline() -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=200,
                        class_weight="balanced",
                        random_state=0,
                    ),
                ),
            ]
        )

    @staticmethod
    def _synthetic_dataset(n: int = 1500, seed: int = 123) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic training data with realistic directional relationships."""

        rng = np.random.default_rng(seed)

        temperature = rng.normal(loc=27.0, scale=7.0, size=n).clip(-5, 45)
        humidity = rng.normal(loc=60.0, scale=18.0, size=n).clip(10, 100)

        # Workforce availability: 0..1
        workforce = rng.beta(a=5, b=2, size=n).clip(0.05, 0.99)

        # Material strength: MPa-ish (e.g., concrete 20–60)
        material_strength = rng.normal(loc=35.0, scale=8.0, size=n).clip(15, 70)

        # Equipment reliability: 0..1
        equip = rng.beta(a=6, b=2.5, size=n).clip(0.05, 0.99)

        # Risk drivers (logit model):
        # - extreme temperatures increase risk
        # - higher humidity increases risk
        # - lower workforce availability increases risk
        # - lower strength increases risk
        # - lower equipment reliability increases risk
        temp_extreme = np.abs(temperature - 24.0)

        logit = (
            -1.2
            + 0.045 * (humidity - 55.0)
            + 0.11 * (temp_extreme - 6.0)
            - 2.2 * (workforce - 0.7)
            - 0.06 * (material_strength - 35.0)
            - 2.0 * (equip - 0.75)
            + rng.normal(0.0, 0.35, size=n)
        )

        p = 1.0 / (1.0 + np.exp(-logit))
        y = rng.binomial(1, p, size=n)

        X = pd.DataFrame(
            {
                "temperature": temperature,
                "humidity": humidity,
                "workforce_availability": workforce,
                "material_strength": material_strength,
                "equipment_reliability": equip,
            }
        )
        return X, y

    @classmethod
    def train_synthetic(cls) -> "RiskPredictor":
        X, y = cls._synthetic_dataset()
        pipe = cls.build_default_pipeline()
        pipe.fit(X[RISK_FEATURES], y)
        return cls(pipe)

    def predict(self, features: dict[str, float]) -> RiskPrediction:
        row = pd.DataFrame([features])[RISK_FEATURES]
        proba = float(self.pipeline.predict_proba(row)[0, 1])
        label = "High" if proba >= 0.70 else ("Medium" if proba >= 0.45 else "Low")
        return RiskPrediction(probability_delay=proba, label=label)

    def explain(self, features: dict[str, float]) -> pd.DataFrame:
        """Explain a single prediction via local logit contributions.

        For logistic regression with standardization:
        logit = intercept + sum_i coef_i * z_i
        where z_i are standardized feature values.

        Returns a DataFrame with per-feature contribution and direction.
        """

        row = pd.DataFrame([features])[RISK_FEATURES]

        scaler: StandardScaler = self.pipeline.named_steps["scaler"]
        clf: LogisticRegression = self.pipeline.named_steps["clf"]

        z = scaler.transform(row)
        coef = clf.coef_.reshape(-1)
        contrib = (z.reshape(-1) * coef).reshape(-1)

        df = pd.DataFrame(
            {
                "feature": RISK_FEATURES,
                "value": [float(row.iloc[0][f]) for f in RISK_FEATURES],
                "z_score": [float(v) for v in z.reshape(-1)],
                "coef": [float(c) for c in coef],
                "logit_contribution": [float(c) for c in contrib],
            }
        )
        df["impact"] = np.where(df["logit_contribution"] >= 0, "increases risk", "reduces risk")
        df = df.sort_values("logit_contribution", ascending=False).reset_index(drop=True)
        return df

    def save(self, path=RISK_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(self.pipeline, path)

    @classmethod
    def load(cls, path=RISK_MODEL_PATH) -> "RiskPredictor":
        pipeline = load(path)
        return cls(pipeline)

    @classmethod
    def load_or_train(cls, path=RISK_MODEL_PATH) -> "RiskPredictor":
        try:
            return cls.load(path)
        except Exception:
            model = cls.train_synthetic()
            model.save(path)
            return model

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "LogisticRegression",
            "features": list(RISK_FEATURES),
        }
