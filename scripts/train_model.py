"""Optional training script.

The Streamlit app auto-trains a model if none exists, but this script lets you
create the model artifact explicitly.

Usage:
  python scripts/train_model.py
"""

from __future__ import annotations

from construct_ai.risk_model import RiskPredictor


def main() -> None:
    model = RiskPredictor.train_synthetic()
    model.save()
    print("Saved model to models/risk_model.joblib")


if __name__ == "__main__":
    main()
