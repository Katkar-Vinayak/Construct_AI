"""CONSTRUCT-AI Streamlit App.

Digital Twin Dashboard that ties together:
- Generative design engine
- Risk prediction model (explainable)
- IoT sensor simulator
- Adaptive decision engine
- Learning/history system

Run:
  streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
import time

import pandas as pd
import streamlit as st

from construct_ai.config import HISTORY_CSV_PATH, RISK_FEATURES
from construct_ai.decision_engine import adaptive_decisions
from construct_ai.generative_design import DesignInputs, best_design, generate_design_options, options_to_frame, weather_bucket
from construct_ai.iot_simulator import SensorSimulator
from construct_ai.learning_system import append_history, compute_learning_insights, load_history
from construct_ai.risk_model import RiskPredictor
from construct_ai.visualizations import (
    cost_time_scatter,
    design_comparison_chart,
    explanation_contributions_chart,
)


def _init_session_state() -> None:
    if "sensor_sim" not in st.session_state:
        st.session_state.sensor_sim = SensorSimulator()
    if "sensor_log" not in st.session_state:
        st.session_state.sensor_log = []  # list[dict]
    if "design_df" not in st.session_state:
        st.session_state.design_df = None
    if "best_design" not in st.session_state:
        st.session_state.best_design = None


def _record_sensor(reading) -> None:
    st.session_state.sensor_log.append(
        {
            "timestamp": reading.timestamp.isoformat(),
            "temperature": reading.temperature,
            "humidity": reading.humidity,
            "vibration": reading.vibration,
            "structural_load": reading.structural_load,
        }
    )
    # Keep only recent window
    st.session_state.sensor_log = st.session_state.sensor_log[-120:]


def sidebar_inputs() -> tuple[DesignInputs, dict[str, float], dict[str, float]]:
    st.sidebar.header("Project Inputs")

    project_name = st.sidebar.text_input("Project name", value="Demo Site A")
    budget_usd = st.sidebar.number_input("Budget (USD)", min_value=200_000, max_value=50_000_000, value=2_500_000, step=50_000)

    material_quality = st.sidebar.slider("Material quality (0–10)", 0.0, 10.0, 6.5, 0.1)
    labor_efficiency = st.sidebar.slider("Labor efficiency (0–10)", 0.0, 10.0, 6.0, 0.1)
    weather_severity = st.sidebar.slider("Weather severity (0–10)", 0.0, 10.0, 4.0, 0.1)

    st.sidebar.divider()
    st.sidebar.subheader("Risk model inputs")

    # Start risk inputs near plausible values.
    temperature = st.sidebar.slider("Temperature (°C)", -5.0, 45.0, 28.0, 0.1)
    humidity = st.sidebar.slider("Humidity (%RH)", 10.0, 100.0, 60.0, 0.5)

    workforce_availability = st.sidebar.slider("Workforce availability (0–1)", 0.1, 1.0, 0.75, 0.01)
    material_strength = st.sidebar.slider("Material strength (MPa)", 15.0, 70.0, 35.0, 0.5)
    equipment_reliability = st.sidebar.slider("Equipment reliability (0–1)", 0.1, 1.0, 0.8, 0.01)

    design_inputs = DesignInputs(
        budget_usd=float(budget_usd),
        material_quality=float(material_quality),
        labor_efficiency=float(labor_efficiency),
        weather_severity=float(weather_severity),
    )

    risk_features = {
        "temperature": float(temperature),
        "humidity": float(humidity),
        "workforce_availability": float(workforce_availability),
        "material_strength": float(material_strength),
        "equipment_reliability": float(equipment_reliability),
    }

    meta = {
        "project_name": project_name,
    }

    return design_inputs, risk_features, meta


def page_dashboard(design_inputs: DesignInputs, risk_features: dict[str, float], meta: dict[str, float]) -> None:
    st.title("CONSTRUCT-AI — Digital Twin Dashboard")
    st.caption("Generative design + risk prediction + sensor simulation + adaptive decisions")

    model = RiskPredictor.load_or_train()

    # Simulate one sensor tick to keep the dashboard alive.
    st.session_state.sensor_sim.weather_severity = design_inputs.weather_severity
    reading = st.session_state.sensor_sim.step()
    _record_sensor(reading)

    # Optionally override risk inputs from live sensors for temperature/humidity.
    use_live_env = st.toggle("Use live sensor temperature/humidity for risk model", value=True)
    effective_features = dict(risk_features)
    if use_live_env:
        effective_features["temperature"] = reading.temperature
        effective_features["humidity"] = reading.humidity

    pred = model.predict(effective_features)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weather", weather_bucket(design_inputs.weather_severity))
    c2.metric("Delay risk", f"{pred.probability_delay:.0%}", pred.label)
    c3.metric("Temp (°C)", f"{reading.temperature:.1f}")
    c4.metric("Humidity (%RH)", f"{reading.humidity:.0f}")

    st.divider()

    left, right = st.columns([0.62, 0.38])

    with left:
        st.subheader("Live Sensors (last ~2 minutes)")
        sensor_df = pd.DataFrame(st.session_state.sensor_log)
        if not sensor_df.empty:
            sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
            sensor_df = sensor_df.set_index("timestamp")
            st.line_chart(sensor_df[["temperature", "humidity"]])
            st.line_chart(sensor_df[["vibration", "structural_load"]])

    with right:
        st.subheader("Risk Alerts")
        if pred.label == "High":
            st.error("High delay probability — activate mitigation plan")
        elif pred.label == "Medium":
            st.warning("Moderate delay probability — add contingency")
        else:
            st.success("Low delay probability — proceed")

        st.subheader("AI Decisions")
        decisions = adaptive_decisions(
            risk_probability=pred.probability_delay,
            temperature_c=float(effective_features["temperature"]),
            humidity_pct=float(effective_features["humidity"]),
            vibration_mms=float(reading.vibration),
            structural_load_kn=float(reading.structural_load),
            workforce_availability=float(effective_features["workforce_availability"]),
        )
        for d in decisions:
            box = st.container(border=True)
            box.markdown(f"**{d.severity}** — {d.title}")
            box.caption(d.rationale)
            box.write(d.recommended_action)

    st.divider()

    st.subheader("Best Design Recommendation")
    if st.session_state.design_df is None:
        st.info("Generate design options in the 'Generative Design' page.")
    else:
        bd = st.session_state.best_design
        st.write(
            {
                "design_id": bd["design_id"],
                "system": bd["system"],
                "foundation": bd["foundation"],
                "primary_structure": bd["primary_structure"],
                "estimated_cost_usd": bd["estimated_cost_usd"],
                "estimated_duration_days": bd["estimated_duration_days"],
                "durability_index": bd["durability_index"],
            }
        )

    st.divider()

    st.subheader("System Status")
    s1, s2, s3 = st.columns(3)
    s1.metric("ML model", "Ready")
    s2.metric("Sensor stream", "Active")
    s3.metric("Learning store", "CSV")

    with st.expander("Save this run to history"):
        if st.button("Append to history", type="primary"):
            decision_summary = "; ".join([d.title for d in decisions])
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "project_name": meta["project_name"],
                "budget_usd": design_inputs.budget_usd,
                "material_quality": design_inputs.material_quality,
                "labor_efficiency": design_inputs.labor_efficiency,
                "weather_severity": design_inputs.weather_severity,
                **{k: effective_features[k] for k in RISK_FEATURES},
                "selected_design_id": (st.session_state.best_design or {}).get("design_id"),
                "design_system": (st.session_state.best_design or {}).get("system"),
                "estimated_cost_usd": (st.session_state.best_design or {}).get("estimated_cost_usd"),
                "estimated_duration_days": (st.session_state.best_design or {}).get("estimated_duration_days"),
                "durability_index": (st.session_state.best_design or {}).get("durability_index"),
                "risk_probability": pred.probability_delay,
                "ai_decision_summary": decision_summary,
                "actual_delay_days": None,
                "notes": "auto-saved from dashboard",
            }
            append_history(record)
            st.success("Saved to data/history.csv")


def page_generative_design(design_inputs: DesignInputs) -> None:
    st.title("Generative Design Engine")
    st.caption("Generate multiple conceptual design options and score cost / durability / time.")

    n_options = st.slider("Number of options", min_value=4, max_value=12, value=8)

    if st.button("Generate designs", type="primary"):
        options = generate_design_options(design_inputs, n_options=n_options)
        df = options_to_frame(options)

        st.session_state.design_df = df
        st.session_state.best_design = df.iloc[0].to_dict()

    if st.session_state.design_df is None:
        st.info("Click 'Generate designs' to create options.")
        return

    df = st.session_state.design_df

    st.subheader("Design Comparison")
    st.altair_chart(design_comparison_chart(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cost vs Time")
        st.altair_chart(cost_time_scatter(df), use_container_width=True)

    with c2:
        st.subheader("Top Recommendation")
        top = df.iloc[0]
        st.metric("Design", f"{top['design_id']} — {top['system']}")
        st.metric("Estimated cost", f"${top['estimated_cost_usd']:,.0f}")
        st.metric("Estimated duration", f"{top['estimated_duration_days']:.1f} days")
        st.metric("Durability index", f"{top['durability_index']:.1f}")

    st.subheader("All Options")
    st.dataframe(
        df[[
            "design_id",
            "system",
            "foundation",
            "primary_structure",
            "prefabrication_level",
            "estimated_cost_usd",
            "estimated_duration_days",
            "durability_index",
            "score_total",
        ]],
        use_container_width=True,
        hide_index=True,
    )


def page_risk_and_sensors(risk_features: dict[str, float], design_inputs: DesignInputs) -> None:
    st.title("Risk Prediction + IoT Sensors")

    model = RiskPredictor.load_or_train()

    st.subheader("Sensor Simulator")
    st.caption("Click to advance the simulated sensors; the dashboard keeps a rolling window.")

    c1, c2, c3 = st.columns([0.22, 0.22, 0.56])
    with c1:
        if st.button("Step sensors"):
            st.session_state.sensor_sim.weather_severity = design_inputs.weather_severity
            reading = st.session_state.sensor_sim.step()
            _record_sensor(reading)

    with c2:
        live_seconds = st.number_input("Live run (seconds)", min_value=3, max_value=30, value=10, step=1)
        run_live = st.button("Run live simulation")

        if st.button("Reset sensor log"):
            st.session_state.sensor_log = []

    with c3:
        st.write("Weather bucket:", weather_bucket(design_inputs.weather_severity))

    sensor_df = pd.DataFrame(st.session_state.sensor_log)
    if not sensor_df.empty:
        sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
        st.dataframe(sensor_df.tail(10), use_container_width=True, hide_index=True)

    if run_live:
        st.session_state.sensor_sim.weather_severity = design_inputs.weather_severity
        status = st.empty()
        kpis = st.empty()
        charts = st.empty()
        prog = st.progress(0)

        for t in range(int(live_seconds)):
            reading = st.session_state.sensor_sim.step()
            _record_sensor(reading)

            sensor_df = pd.DataFrame(st.session_state.sensor_log)
            if not sensor_df.empty:
                sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])

            effective = dict(risk_features)
            effective["temperature"] = reading.temperature
            effective["humidity"] = reading.humidity

            pred = model.predict(effective)

            status.info(f"Live tick {t + 1}/{int(live_seconds)} — delay risk: {pred.probability_delay:.0%} ({pred.label})")

            with kpis.container():
                a, b, c, d = st.columns(4)
                a.metric("Temp (°C)", f"{reading.temperature:.1f}")
                b.metric("Humidity (%RH)", f"{reading.humidity:.0f}")
                c.metric("Vibration (mm/s)", f"{reading.vibration:.2f}")
                d.metric("Load (kN)", f"{reading.structural_load:.0f}")

            with charts.container():
                if not sensor_df.empty:
                    tmp = sensor_df.set_index("timestamp")
                    st.line_chart(tmp[["temperature", "humidity"]])
                    st.line_chart(tmp[["vibration", "structural_load"]])

            prog.progress(int((t + 1) / int(live_seconds) * 100))
            time.sleep(1)

        st.success("Live simulation complete.")

    st.divider()

    st.subheader("Risk Prediction")
    use_latest = st.toggle("Use latest sensor temperature/humidity", value=True)

    effective = dict(risk_features)
    if use_latest and not sensor_df.empty:
        effective["temperature"] = float(sensor_df.iloc[-1]["temperature"])
        effective["humidity"] = float(sensor_df.iloc[-1]["humidity"])

    pred = model.predict(effective)
    st.metric("Predicted delay probability", f"{pred.probability_delay:.0%}", pred.label)

    explain_df = model.explain(effective)
    st.subheader("Explainable prediction")
    st.caption("Bars show each feature's contribution to the model's risk score (logit space).")
    st.altair_chart(explanation_contributions_chart(explain_df), use_container_width=True)
    st.dataframe(explain_df, use_container_width=True, hide_index=True)


def page_learning() -> None:
    st.title("Learning From Past Projects")

    history = load_history()
    insights = compute_learning_insights(history)

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", str(insights.n_records))
    c2.metric("Labeled outcomes", str(insights.n_labeled))
    c3.metric("Suggested buffer", f"{insights.recommended_schedule_buffer_pct:.1f}%")

    st.subheader("Learned insights")
    for d in insights.common_delay_drivers:
        st.write("-", d)

    st.info(f"Suggested extra curing time: {insights.recommended_extra_curing_days:.1f} days")

    st.divider()

    st.subheader("History table")
    st.dataframe(history.tail(50), use_container_width=True)

    st.divider()

    st.subheader("Add a completed project outcome")
    st.caption("Add a ground-truth delay to improve the learning insights.")

    if history.empty:
        st.warning("No saved runs yet. Save a run from the Dashboard first.")
        return

    last = history.tail(1).iloc[0]
    with st.form("label_form"):
        row_index = st.number_input("Row index to label (0-based)", min_value=0, max_value=max(0, len(history) - 1), value=max(0, len(history) - 1))
        actual_delay_days = st.number_input("Actual delay days", min_value=0.0, max_value=365.0, value=0.0, step=1.0)
        notes = st.text_input("Notes", value="labeled outcome")
        submitted = st.form_submit_button("Save label")

    if submitted:
        history.loc[int(row_index), "actual_delay_days"] = float(actual_delay_days)
        history.loc[int(row_index), "notes"] = notes
        history.to_csv(HISTORY_CSV_PATH, index=False)
        st.success("Saved outcome label.")
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="CONSTRUCT-AI", layout="wide")
    _init_session_state()

    design_inputs, risk_features, meta = sidebar_inputs()

    page = st.sidebar.radio(
        "Navigation",
        options=["Dashboard", "Generative Design", "Risk + Sensors", "Learning"],
        index=0,
    )

    if page == "Dashboard":
        page_dashboard(design_inputs, risk_features, meta)
    elif page == "Generative Design":
        page_generative_design(design_inputs)
    elif page == "Risk + Sensors":
        page_risk_and_sensors(risk_features, design_inputs)
    else:
        page_learning()


if __name__ == "__main__":
    main()
