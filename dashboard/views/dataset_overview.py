"""Data Overview — what the raw data looks like before any engineering."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import load_raw
from theme import (
    CARD_BORDER,
    ERROR,
    PRIMARY,
    SECONDARY,
    TEXT_MUTED,
    TEXT_PRIMARY,
    page_header,
    style_fig,
    takeaway,
    what_this_means,
)

raw = load_raw()

page_header("Data Overview", "The raw Titanic passenger manifest used to train and validate the model.")

# Bold, high-contrast fills for the four summary metric boxes only —
# solid saturated backgrounds with white text instead of pastel tints.
st.markdown(
    """
    <style>
        .st-key-metric-row-top [data-testid="stColumn"]:nth-of-type(1) [data-testid="stMetric"] {
            background-color: #2563EB !important;
        }
        .st-key-metric-row-top [data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetric"] {
            background-color: #10B981 !important;
        }
        .st-key-metric-row-top [data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetric"] {
            background-color: #F59E0B !important;
        }
        .st-key-metric-row-top [data-testid="stColumn"]:nth-of-type(4) [data-testid="stMetric"] {
            background-color: #8B5CF6 !important;
        }
        .st-key-metric-row-top [data-testid="stMetric"] [data-testid="stMetricLabel"],
        .st-key-metric-row-top [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: white !important;
        }
        div[class*="st-key-card-core-features"] {
            background-color: #06B6D4 !important;
        }
        div[class*="st-key-card-engineered-features"] {
            background-color: #8B5CF6 !important;
        }
        div[class*="st-key-card-core-features"] *,
        div[class*="st-key-card-engineered-features"] * {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="metric-row-top"):
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
    # Outside text labels sit past the bar end — without headroom, the
    # longest bar's label (Cabin, ~77%) gets clipped by the plot's right edge.
    fig.update_xaxes(range=[0, missing_pct.max() * 1.15])
    st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
with summary_col:
    n_incomplete = int(has_missing.sum())
    top_feature, top_pct = missing_pct.index[0], missing_pct.iloc[0]
    completeness = 100 - (raw.isna().sum().sum() / raw.size * 100)
    with st.container(border=True, height=380, key="card-missing-summary"):
        st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Features with Missing Values</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.4rem; font-weight:700; color:{TEXT_PRIMARY}; margin-bottom:1rem;'>{n_incomplete}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Most Incomplete Feature</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.4rem; font-weight:700; color:{TEXT_PRIMARY}; margin-bottom:1rem;'>{top_feature} ({top_pct:.0f}%)</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Overall Dataset Completeness</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:1.4rem; font-weight:700; color:{TEXT_PRIMARY};'>{completeness:.1f}%</div>", unsafe_allow_html=True)

takeaway(
    "Only a couple of features have missing data, so we can fix those specific gaps "
    "without touching the rest of the dataset."
)
what_this_means(
    observation="Cabin is missing for 77% of passengers, and Age is missing for about 20%.",
    impact="Cabin is missing too often to fill in with a guess, but knowing whether a cabin "
    "was recorded at all still tells us something.",
    decision="We turned Cabin into a simple yes or no flag and filled in the missing Age "
    "values instead of dropping them.",
)

st.divider()
st.subheader("Target Distribution")
counts = raw["Survived"].value_counts().sort_index()
labels = ["Did Not Survive", "Survived"]
pct_values = counts.values / counts.sum() * 100

with st.container(border=True, key="card-target-dist"):
    _, chart_col, _ = st.columns([1, 4, 1])
    with chart_col:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=counts.values,
            marker_color=[ERROR, PRIMARY],
            marker_cornerradius=8,
            text=[f"{c:,}" for c in counts.values],
            textposition="outside",
            textfont=dict(size=15, color=TEXT_PRIMARY),
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
            showgrid=True, gridcolor="#F1F5F9", zeroline=False,
            # Headroom for the outside count label above the taller bar —
            # without it, the label gets clipped by the plot's top edge.
            range=[0, counts.values.max() * 1.2],
        )
        st.plotly_chart(fig, use_container_width=True)

takeaway(
    "The dataset has more passengers who did not survive than those who survived, but "
    "the difference is not large enough to cause major issues."
)
what_this_means(
    observation="About 62% of passengers did not survive, and 38% did.",
    impact="With this kind of imbalance, a model could look 62% accurate just by guessing "
    "\"did not survive\" every time.",
    decision="That's why we always check F1 score and ROC-AUC alongside accuracy instead "
    "of relying on accuracy alone.",
)

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
    with st.container(border=True, key="card-core-features"):
        st.markdown("**Core Features**")
        st.markdown(
            "- Passenger Class\n"
            "- Sex\n"
            "- Age\n"
            "- Fare\n"
            "- Embarked"
        )
with eng_col:
    with st.container(border=True, key="card-engineered-features"):
        st.markdown("**Engineered Features**")
        st.markdown(
            "- Family Size\n"
            "- Cabin Presence\n"
            "- Passenger Title"
        )
takeaway(
    "Every feature in this set earned its place through testing, not guesswork, giving us "
    "a small model that's still easy to explain."
)
