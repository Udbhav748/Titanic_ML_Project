"""Project Architecture — how a prediction moves through the system."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import PRIMARY, QUATERNARY, SUCCESS, TERTIARY, WARNING, highlight_card, page_header

page_header("Project Architecture", "How a prediction moves through the system.")

# Five-color rotation reused across boxes, chips, and Pipeline Components —
# solid, saturated fills instead of pastel tints, same box model everywhere
# (size, padding, radius untouched).
PALETTE = [PRIMARY, SUCCESS, WARNING, QUATERNARY, TERTIARY]

st.markdown(
    f"""
    <style>
        .arch-step {{
            border-radius: 8px;
            padding: 0.5rem 1rem;
            text-align: center;
            font-weight: 700;
            color: white;
            max-width: 420px;
            margin: 0 auto;
        }}
        .arch-arrow {{
            text-align: center;
            color: #94A3B8;
            font-size: 1.2rem;
            line-height: 1.6rem;
        }}
        .arch-chip {{
            display: inline-block;
            border-radius: 12px;
            padding: 0.15rem 0.7rem;
            margin: 0.2rem;
            font-size: 0.78rem;
            font-weight: 600;
        }}
        div[class~="st-key-card-highlight-input"] {{ border-top: 4px solid {PRIMARY} !important; }}
        div[class~="st-key-card-highlight-preprocessing"] {{ border-top: 4px solid {SUCCESS} !important; }}
        div[class~="st-key-card-highlight-machine-learning-model"] {{ border-top: 4px solid {WARNING} !important; }}
        div[class~="st-key-card-highlight-prediction-engine"] {{ border-top: 4px solid {QUATERNARY} !important; }}
        div[class~="st-key-card-highlight-decision-intelligence"] {{ border-top: 4px solid {TERTIARY} !important; }}
        div[class~="st-key-card-highlight-dashboard"] {{ border-top: 4px solid {PRIMARY} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_steps(steps: list[str], start: int) -> int:
    for i, step in enumerate(steps):
        accent = PALETTE[(start + i) % len(PALETTE)]
        st.markdown(
            f"<div class='arch-step' style='background-color:{accent}; "
            f"box-shadow:0 3px 10px {accent}66;'>{step}</div>",
            unsafe_allow_html=True,
        )
        if i < len(steps) - 1:
            st.markdown("<div class='arch-arrow'>&#8595;</div>", unsafe_allow_html=True)
    return start + len(steps)


def render_step_with_subitems(title: str, subitems: list[str], color_index: int) -> None:
    """A flow step whose box is annotated with the smaller capabilities inside it."""
    accent = PALETTE[color_index % len(PALETTE)]
    st.markdown(
        f"<div class='arch-step' style='background-color:{accent}; "
        f"box-shadow:0 3px 10px {accent}66;'>{title}</div>",
        unsafe_allow_html=True,
    )
    chips = "".join(
        f"<span class='arch-chip' style='background-color:white; border:1.5px solid {accent}; color:{accent};'>{s}</span>"
        for s in subitems
    )
    st.markdown(f"<div style='text-align:center; margin-top:0.3rem;'>{chips}</div>", unsafe_allow_html=True)


def arrow() -> None:
    st.markdown("<div class='arch-arrow'>&#8595;</div>", unsafe_allow_html=True)


# ============================================================
# 1. System Overview
# ============================================================
st.subheader("System Overview")
idx = render_steps(["User", "Passenger Information", "Input Validation"], 0)
arrow()
render_step_with_subitems(
    "Production Preprocessing Pipeline",
    ["Missing Value Handling", "Feature Engineering", "Feature Validation"],
    color_index=idx,
)
idx += 1
arrow()
idx = render_steps(["Machine Learning Model"], idx)
arrow()
render_step_with_subitems(
    "Prediction Engine",
    ["Probability", "Confidence", "Feature Contributions"],
    color_index=idx,
)
idx += 1
arrow()
render_steps(["Decision Intelligence Panel", "Interactive Dashboard"], idx)

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
        highlight_card(icon, title, text, show_icon=False)

# ============================================================
# 3. Data Flow
# ============================================================
st.write("")
st.divider()
st.subheader("Data Flow")
render_steps(["Raw Input", "Validation", "Transformation", "Prediction", "Interpretation", "Visualization"], 0)
