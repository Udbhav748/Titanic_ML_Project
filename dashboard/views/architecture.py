"""Project Architecture — how a prediction moves through the system."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import BACKGROUND, CARD_BORDER, TEXT_MUTED, highlight_card, page_header, render_flow

page_header("Project Architecture", "How a prediction moves through the system.")


def render_step_with_subitems(title: str, subitems: list[str]) -> None:
    """A flow step whose box is annotated with the smaller capabilities inside it."""
    st.markdown(f"<div class='flow-step'>{title}</div>", unsafe_allow_html=True)
    chips = "".join(
        f"<span style='display:inline-block; background:{BACKGROUND}; border:1px solid {CARD_BORDER}; "
        f"border-radius:12px; padding:0.15rem 0.7rem; margin:0.2rem; font-size:0.78rem; color:{TEXT_MUTED};'>{s}</span>"
        for s in subitems
    )
    st.markdown(f"<div style='text-align:center; margin-top:0.3rem;'>{chips}</div>", unsafe_allow_html=True)


def arrow() -> None:
    st.markdown("<div class='flow-arrow'>&#8595;</div>", unsafe_allow_html=True)


# ============================================================
# 1. System Overview
# ============================================================
st.subheader("System Overview")
render_flow(["User", "Passenger Information", "Input Validation"])
arrow()
render_step_with_subitems(
    "Production Preprocessing Pipeline",
    ["Missing Value Handling", "Feature Engineering", "Feature Validation"],
)
arrow()
render_flow(["Machine Learning Model"])
arrow()
render_step_with_subitems(
    "Prediction Engine",
    ["Probability", "Confidence", "Feature Contributions"],
)
arrow()
render_flow(["Decision Intelligence Panel", "Interactive Dashboard"])

# ============================================================
# 2. Pipeline Components
# ============================================================
st.write("")
st.divider()
st.subheader("Pipeline Components")
COMPONENTS = [
    ("input", "Input", "Collects and validates passenger information from the user."),
    ("tune", "Preprocessing", "Transforms raw passenger data into the production feature set."),
    ("model_training", "Machine Learning Model", "Predicts survival probability from the processed features."),
    ("insights", "Prediction Engine", "Calculates confidence, decision strength, and feature contributions."),
    ("hub", "Decision Intelligence", "Explores counterfactuals, sensitivity, and similar historical cases."),
    ("dashboard", "Dashboard", "Displays results and interactive analytics."),
]
c_row1 = st.columns(3)
c_row2 = st.columns(3)
for i, (icon, title, text) in enumerate(COMPONENTS):
    col = c_row1[i] if i < 3 else c_row2[i - 3]
    with col:
        highlight_card(icon, title, text)

# ============================================================
# 3. Data Flow
# ============================================================
st.write("")
st.divider()
st.subheader("Data Flow")
render_flow(["Raw Input", "Validation", "Transformation", "Prediction", "Interpretation", "Visualization"])
