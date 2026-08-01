"""Data Overview — what the raw data looks like before any engineering."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import load_raw
from theme import CARD_BORDER, ERROR, PRIMARY, SECONDARY, TEXT_MUTED, page_header, style_fig, takeaway

raw = load_raw()

page_header("Data Overview", "The raw Titanic passenger manifest used to train and validate the model.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{raw.shape[0]:,}")
c2.metric("Columns", raw.shape[1])
c3.metric("Duplicate Rows", int(raw.duplicated().sum()))
c4.metric("Survival Rate", f"{raw['Survived'].mean():.1%}")

st.divider()
st.subheader("Dataset Preview")
st.dataframe(raw.head(10), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Missing Data Overview")
missing_pct = (raw.isna().mean() * 100).sort_values(ascending=False)
has_missing = missing_pct > 0
bar_colors = [ERROR if m else SECONDARY for m in has_missing]

chart_col, summary_col = st.columns([2, 1])
with chart_col:
    fig = px.bar(
        x=missing_pct.values, y=missing_pct.index, orientation="h",
        labels={"x": "% missing", "y": ""},
        text=[f"{v:.0f}%" for v in missing_pct.values],
    )
    fig.update_traces(marker_color=bar_colors, textposition="outside")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
with summary_col:
    n_incomplete = int(has_missing.sum())
    top_feature, top_pct = missing_pct.index[0], missing_pct.iloc[0]
    completeness = 100 - (raw.isna().sum().sum() / raw.size * 100)
    with st.container(border=True, height=380):
        st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Features with Missing Values</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.4rem; font-weight:700; color:#14213D; margin-bottom:1rem;'>{n_incomplete}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Most Incomplete Feature</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.4rem; font-weight:700; color:#14213D; margin-bottom:1rem;'>{top_feature} ({top_pct:.0f}%)</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Overall Dataset Completeness</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.4rem; font-weight:700; color:#14213D;'>{completeness:.1f}%</div>", unsafe_allow_html=True)

takeaway(
    "Missing values are concentrated in only a small number of features, enabling targeted "
    "preprocessing without affecting the overall dataset quality."
)

st.divider()
st.subheader("Target Distribution")
counts = raw["Survived"].value_counts().sort_index()
labels = ["Did Not Survive", "Survived"]
pct_values = counts.values / counts.sum() * 100

with st.container(border=True):
    _, chart_col, _ = st.columns([1, 4, 1])
    with chart_col:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=counts.values,
            marker_color=[ERROR, PRIMARY],
            marker_cornerradius=8,
            text=[f"{c:,}" for c in counts.values],
            textposition="outside",
            textfont=dict(size=15, color="#14213D"),
            hovertemplate="<b>%{x}</b><br>%{y:,} passengers<extra></extra>",
        ))
        for label, count, pct in zip(labels, counts.values, pct_values):
            fig.add_annotation(
                x=label, y=count / 2, text=f"{pct:.1f}%",
                showarrow=False, font=dict(size=13, color="white"),
            )
        fig = style_fig(fig, height=320)
        fig.update_layout(
            showlegend=False, bargap=0.5,
            hoverlabel=dict(bgcolor="white", font_size=13, bordercolor=CARD_BORDER),
        )
        fig.update_xaxes(showline=True, linecolor=CARD_BORDER, ticks="", title=None, tickfont=dict(size=14))
        fig.update_yaxes(
            showticklabels=False, title="Passengers", showline=True, linecolor=CARD_BORDER,
            showgrid=True, gridcolor="#F0F0F0", zeroline=False,
        )
        st.plotly_chart(fig, use_container_width=True)

takeaway("The dataset is moderately imbalanced, with approximately 62% of passengers not surviving and 38% surviving.")

st.divider()
st.subheader("Feature Summary")
reference = pd.DataFrame([
    ("Pclass", "Categorical", "Ticket class (1/2/3) — proxy for socioeconomic status"),
    ("Sex", "Categorical", "Passenger sex"),
    ("Age", "Numeric", "Age in years, imputed where missing"),
    ("Fare", "Numeric", "Fare paid, right-skewed"),
    ("Embarked", "Categorical", "Port of embarkation (C/Q/S)"),
    ("SibSp / Parch", "Numeric", "Siblings/spouses and parents/children aboard — combined into Family Size"),
    ("Cabin", "Text", "77% missing — converted to a Has Cabin flag"),
    ("Name", "Text", "Source for the engineered Title feature"),
], columns=["Feature", "Type", "Description"])
st.dataframe(reference, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Features Used for Prediction")
core_col, eng_col = st.columns(2)
with core_col:
    with st.container(border=True):
        st.markdown("**Core Features**")
        st.markdown(
            "- Passenger Class\n"
            "- Sex\n"
            "- Age\n"
            "- Fare\n"
            "- Embarked"
        )
with eng_col:
    with st.container(border=True):
        st.markdown("**Engineered Features**")
        st.markdown(
            "- Family Size\n"
            "- Cabin Presence\n"
            "- Passenger Title"
        )
takeaway(
    "These features were selected through statistical analysis, feature evaluation, and engineering "
    "review to create a compact, interpretable, and effective prediction model."
)
