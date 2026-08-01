"""Entry point — defines navigation for the Titanic Survival Prediction dashboard."""
import streamlit as st
from theme import apply_theme

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

pages = [
    st.Page("views/home.py", title="Home", default=True),
    st.Page("views/dataset_overview.py", title="Data Overview"),
    st.Page("views/eda.py", title="Data Analysis"),
    st.Page("views/model_performance.py", title="Model Performance"),
    st.Page("views/live_prediction.py", title="Live Prediction"),
    st.Page("views/architecture.py", title="Project Architecture"),
]

pg = st.navigation(pages)
pg.run()
