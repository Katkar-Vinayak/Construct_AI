"""Visualization helpers for Streamlit.

We keep charts in a module so the Streamlit app stays readable.
"""

from __future__ import annotations

import altair as alt
import pandas as pd


def design_comparison_chart(design_df: pd.DataFrame) -> alt.Chart:
    """Bar chart comparing total scores."""

    base = alt.Chart(design_df).encode(
        x=alt.X("design_id:N", sort=None, title="Design Option"),
        tooltip=[
            "design_id:N",
            "system:N",
            alt.Tooltip("estimated_cost_usd:Q", format=",.0f"),
            alt.Tooltip("estimated_duration_days:Q", format=".1f"),
            alt.Tooltip("durability_index:Q", format=".1f"),
            alt.Tooltip("score_total:Q", format=".1f"),
        ],
    )

    bars = base.mark_bar().encode(
        y=alt.Y("score_total:Q", title="Total Score"),
        color=alt.Color("system:N", legend=alt.Legend(title="System")),
    )

    return bars.properties(height=260)


def cost_time_scatter(design_df: pd.DataFrame) -> alt.Chart:
    """Cost vs time scatter with durability size."""

    return (
        alt.Chart(design_df)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X("estimated_duration_days:Q", title="Estimated Duration (days)"),
            y=alt.Y("estimated_cost_usd:Q", title="Estimated Cost (USD)"),
            size=alt.Size("durability_index:Q", title="Durability", scale=alt.Scale(range=[60, 520])),
            color=alt.Color("system:N", legend=alt.Legend(title="System")),
            tooltip=[
                "design_id:N",
                "system:N",
                alt.Tooltip("estimated_cost_usd:Q", format=",.0f"),
                alt.Tooltip("estimated_duration_days:Q", format=".1f"),
                alt.Tooltip("durability_index:Q", format=".1f"),
            ],
        )
        .properties(height=260)
    )


def explanation_contributions_chart(explain_df: pd.DataFrame) -> alt.Chart:
    """Horizontal bar chart of logit contributions."""

    df = explain_df.copy()
    df["feature"] = df["feature"].str.replace("_", " ")

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            x=alt.X("logit_contribution:Q", title="Contribution to delay risk (logit)"),
            color=alt.condition(
                alt.datum.logit_contribution >= 0,
                alt.value("#d62728"),
                alt.value("#2ca02c"),
            ),
            tooltip=[
                "feature:N",
                alt.Tooltip("value:Q", format=".2f"),
                alt.Tooltip("logit_contribution:Q", format=".3f"),
                "impact:N",
            ],
        )
        .properties(height=220)
    )
