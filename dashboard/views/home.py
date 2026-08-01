"""Home — landing page for the project."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import evaluate_v2_on_test, load_feature_schema, load_raw, load_stage4_results
from theme import highlight_card, metric_card, render_flow

raw = load_raw()
schema = load_feature_schema()
results = load_stage4_results()
_, _, _, _, metrics = evaluate_v2_on_test()

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
row1 = st.columns(4)
row2 = st.columns(4)
with row1[0]:
    metric_card("Dataset", f"{len(raw):,} Passengers")
with row1[1]:
    metric_card("Models Compared", str(len(results)))
with row1[2]:
    metric_card("Production Algorithm", "Logistic Regression")
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
h_row1 = st.columns(3)
h_row2 = st.columns(3)
for i, (icon, title, text) in enumerate(HIGHLIGHTS):
    col = h_row1[i] if i < 3 else h_row2[i - 3]
    with col:
        highlight_card(icon, title, text)

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
        with st.container(border=True):
            st.page_link(path, label=label, icon=f":material/{icon}:", use_container_width=True)
