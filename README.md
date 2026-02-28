# CONSTRUCT-AI (Hackathon Demo)

CONSTRUCT-AI simulates an **AI-driven generative design** + **risk prediction** + **digital twin** workflow for construction projects.

It includes:
- Generative design engine (multiple design options + scoring)
- Risk prediction model (scikit-learn, explainable)
- IoT sensor simulator (temperature, humidity, vibration, structural load)
- Adaptive decision engine (human-readable schedule / reinforcement / curing actions)
- Learning system (CSV history + “learn from past projects” insights)

## 1) Quickstart

### Windows (PowerShell)

```powershell
cd "c:\Users\Nikitha\Desktop\LT\construct_ai"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 2) Project structure

- `app.py` – Streamlit digital twin dashboard (main entrypoint)
- `construct_ai/` – Python package with core logic
- `data/history.csv` – stored project history (learning system)
- `models/` – saved ML model (`risk_model.joblib`)
- `scripts/train_model.py` – optional CLI training script

## 3) Notes for judges

- All data is simulated/synthetic (safe for hackathon demos).
- The ML model is intentionally simple but explainable.
- The UI shows a full loop: sensor → risk → decisions → learning.
