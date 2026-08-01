"""Data Analysis — the highest-signal findings, interactive."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import load_transformed
from theme import CARD_BORDER, ERROR, PRIMARY, TEXT_MUTED, page_header, render_flow, style_fig, takeaway

df = load_transformed()

page_header("Data Analysis", "The key patterns that shaped which features power the model.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Feature Distributions", "Survival Analysis", "Correlation Heatmap",
    "Class x Sex Interaction", "VIF Analysis", "Feature Selection Journey",
])

with tab1:
    metric = st.radio("Distribution:", ["Age", "Fare"], horizontal=True)
    fig = px.histogram(
        df, x=metric, color="Survived", barmode="overlay", opacity=0.65,
        color_discrete_map={0: "#C62828", 1: PRIMARY},
        labels={"Survived": "Outcome"},
    )
    fig.for_each_trace(lambda t: t.update(name="Survived" if t.name == "1" else "Did Not Survive"))
    st.plotly_chart(style_fig(fig), use_container_width=True)
    if metric == "Fare":
        takeaway("Fare separates outcomes more cleanly than Age, and is heavily right-skewed — log-transformed inside the model pipeline.")
    else:
        takeaway("Age distributions look similar between groups except for a small child survival spike.")

with tab2:
    filter_col, _ = st.columns([1, 3])
    with filter_col:
        factor = st.selectbox("Break down survival rate by:", ["Sex", "Pclass", "Title", "Embarked"])
    rate = df.groupby(factor, observed=True)["Survived"].mean().sort_values(ascending=False).mul(100)
    overall_rate = df["Survived"].mean() * 100

    with st.container(border=True):
        _, chart_col, _ = st.columns([1, 10, 1])
        with chart_col:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=rate.index.astype(str), y=rate.values,
                marker_color=PRIMARY, marker_cornerradius=8,
                text=[f"{v:.0f}%" for v in rate.values],
                textposition="outside", textfont=dict(size=14, color="#14213D"),
                hovertemplate="<b>%{x}</b><br>Survival rate: %{y:.1f}%<extra></extra>",
            ))
            fig.add_hline(
                y=overall_rate, line_dash="dash", line_color=TEXT_MUTED, line_width=1.5,
                annotation_text=f"Overall average: {overall_rate:.1f}%",
                annotation_position="top right", annotation_font=dict(size=12, color=TEXT_MUTED),
            )
            fig = style_fig(fig, height=340)
            fig.update_layout(
                hoverlabel=dict(bgcolor="white", font_size=13, bordercolor=CARD_BORDER),
                bargap=0.4,
            )
            fig.update_xaxes(title=factor, showline=True, linecolor=CARD_BORDER, ticks="", tickfont=dict(size=13))
            fig.update_yaxes(
                showticklabels=False, title="Survival Rate (%)", showline=True, linecolor=CARD_BORDER,
                showgrid=True, gridcolor="#F0F0F0", zeroline=False,
                range=[0, max(rate.values.max(), overall_rate) * 1.25],
            )
            st.plotly_chart(fig, use_container_width=True)

    takeaway("Sex, Pclass, and Title all separate survival cleanly — the three strongest predictors in the model.")

with tab3:
    corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "FarePerPerson", "HasCabin", "IsAlone"]
    corr = df[corr_cols].corr()
    fig = px.imshow(
        corr.round(2), text_auto=True, color_continuous_scale=["#C62828", "#F0EFEC", PRIMARY],
        zmin=-1, zmax=1, labels={"color": "Pearson r"},
    )
    st.plotly_chart(style_fig(fig, height=480), use_container_width=True)
    takeaway("FamilySize correlates strongly with both SibSp and Parch, and FarePerPerson tracks Fare almost 1:1 — exactly the redundancy VIF quantifies next.")

with tab4:
    heatmap_data = df.groupby(["Pclass", "Sex"], observed=True)["Survived"].mean().unstack().mul(100).round(1)
    fig = px.imshow(
        heatmap_data, text_auto=True, color_continuous_scale=["#F0EFEC", PRIMARY],
        labels={"color": "Survival rate (%)"},
    )
    st.plotly_chart(style_fig(fig, height=350), use_container_width=True)
    takeaway("Sex sets the baseline; class reshapes it unevenly — 1st-class women survive at 97%, even 1st-class men only reach 37%.")

with tab5:
    X = df[["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "FarePerPerson", "HasCabin", "IsAlone"]].copy()
    X["Sex_male"] = (df["Sex"] == "male").astype(int)
    X_const = X.assign(const=1.0)
    vif = pd.Series(
        [variance_inflation_factor(X_const.values, i) for i in range(X.shape[1])],
        index=X.columns,
    ).sort_values(ascending=False)
    display_vif = vif.replace(np.inf, 50)
    colors = [ERROR if v > 5 else PRIMARY for v in vif]
    fig = px.bar(
        x=display_vif.values, y=display_vif.index, orientation="h",
        labels={"x": "VIF (capped at 50 for display)", "y": ""},
        text=["∞" if np.isinf(v) else f"{v:.1f}" for v in vif],
    )
    fig.update_traces(marker_color=colors, textposition="outside")
    fig.add_vline(x=5, line_dash="dash", line_color="#5B6572")
    st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
    takeaway("SibSp, Parch, and FamilySize are perfectly collinear (VIF = infinite) — never train a model with all three together.")

with tab6:
    render_flow([
        "Original Candidate Features (11)",
        "Statistical Significance Testing",
        "Mutual Information Ranking",
        "Multicollinearity Check (VIF)",
        "Engineering Trade-off Review",
        "Final Production Features (8)",
    ])
    st.markdown(
        "- Three features — sibling/spouse count, parent/child count, and family size — were perfectly "
        "collinear; VIF flagged them as infinite\n"
        "- Two more features offered only marginal information gain relative to their engineering cost, "
        "and were cut\n"
        "- The resulting 8-feature schema let a simpler linear model outperform the original ensemble"
    )
    takeaway("Every removed feature was cut on evidence — redundancy or marginal value — not convenience.")
