"""Home — landing page for the project."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import evaluate_v2_on_test, load_feature_schema, load_raw, load_stage4_results
from theme import (
    ERROR,
    PRIMARY,
    QUATERNARY,
    SUCCESS,
    TERTIARY,
    WARNING,
    highlight_card,
    metric_card,
    render_flow,
)

raw = load_raw()
schema = load_feature_schema()
results = load_stage4_results()
_, _, _, _, metrics = evaluate_v2_on_test()

BOLD_PALETTE = [PRIMARY, SUCCESS, WARNING, QUATERNARY, TERTIARY, ERROR]


def _slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-")


def bold_card_css(labels: list[str], key_prefix: str) -> str:
    rules = []
    for i, label in enumerate(labels):
        color = BOLD_PALETTE[i % len(BOLD_PALETTE)]
        rules.append(
            f'div[class*="st-key-card-{key_prefix}-{_slug(label)}"] {{ background-color: {color} !important; }}'
            f'div[class*="st-key-card-{key_prefix}-{_slug(label)}"] * {{ color: white !important; }}'
        )
    return "<style>" + "".join(rules) + "</style>"

# --- Title, one-liner, call to action -----------------------------------
st.title("Titanic Survival Prediction")
st.markdown(
    "An end-to-end machine learning system that predicts Titanic passenger survival "
    "from raw passenger data."
)
st.page_link("views/live_prediction.py", label="Try a Live Prediction", icon=":material/bolt:")

st.divider()

# --- Project Snapshot -----------------------------------------------------
st.subheader("Project Snapshot")
st.markdown(
    bold_card_css(
        ["Dataset", "Models Compared", "Selected Model", "Accuracy",
         "ROC-AUC", "Features Used", "API", "Deployment"],
        "metric",
    ),
    unsafe_allow_html=True,
)
row1 = st.columns(4)
row2 = st.columns(4)
with row1[0]:
    metric_card("Dataset", f"{len(raw):,} Passengers")
with row1[1]:
    metric_card("Models Compared", str(len(results)))
with row1[2]:
    metric_card("Selected Model", "Logistic Regression")
with row1[3]:
    metric_card("Accuracy", f"{metrics['accuracy']:.1%}")
with row2[0]:
    metric_card("ROC-AUC", f"{metrics['roc_auc']:.3f}")
with row2[1]:
    metric_card("Features Used", str(len(schema["production_features"])))
with row2[2]:
    metric_card("API", "FastAPI")
with row2[3]:
    metric_card("Deployment", "Docker + AWS EC2")

st.divider()

# --- Production Pipeline ---------------------------------------------------
st.subheader("Production Pipeline")
render_flow(["Input", "Preprocessing", "Machine Learning Model", "Prediction", "Dashboard"], vertical=False)

st.divider()

# --- Project Highlights -----------------------------------------------------
st.subheader("Project Highlights")
HIGHLIGHTS = [
    ("route", "End-to-End ML Pipeline", "Raw data flows through preprocessing, model training, evaluation, and live prediction."),
    ("query_stats", "Statistical Feature Engineering", "Engineered and validated using statistical testing, mutual information, and VIF analysis."),
    ("fact_check", "Evidence-Based Model Selection", "Several models were compared under one consistent framework to select the final model."),
    ("api", "Production API", "A FastAPI-based prediction service with input validation and consistent inference."),
    ("dashboard", "Interactive Analytics Dashboard", "An interactive dashboard for exploring the dataset, model, and live predictions."),
    ("cloud_done", "Cloud Deployment", "Containerized with Docker and deployed on AWS EC2 for real-time inference."),
]
st.markdown(bold_card_css([title for _, title, _ in HIGHLIGHTS], "highlight"), unsafe_allow_html=True)
h_row1 = st.columns(3)
h_row2 = st.columns(3)
for i, (icon, title, text) in enumerate(HIGHLIGHTS):
    col = h_row1[i] if i < 3 else h_row2[i - 3]
    with col:
        highlight_card(icon, title, text, show_icon=False)

st.divider()

# --- Quick Navigation -----------------------------------------------------
st.subheader("Explore the Project")
NAV_ITEMS = [
    ("views/dataset_overview.py", "Data Overview", "table_chart"),
    ("views/eda.py", "Data Analysis", "monitoring"),
    ("views/model_performance.py", "Model Performance", "model_training"),
    ("views/live_prediction.py", "Live Prediction", "bolt"),
    ("views/architecture.py", "Project Architecture", "account_tree"),
]
n_row1 = st.columns(3)
n_row2 = st.columns(2)
for i, (path, label, icon) in enumerate(NAV_ITEMS):
    col = n_row1[i] if i < 3 else n_row2[i - 3]
    with col:
        with st.container(border=True, key=f"card-nav-{icon}"):
            st.page_link(path, label=label, use_container_width=True)
