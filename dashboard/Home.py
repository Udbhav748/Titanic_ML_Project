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

LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="220" height="33" viewBox="0 0 220 33">
  <text x="0" y="24" font-family="sans-serif" font-size="22" font-weight="800" fill="#111827">Titanic<tspan fill="#2563EB"> Insights</tspan></text>
</svg>
"""
st.logo(LOGO_SVG, size="large")

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
