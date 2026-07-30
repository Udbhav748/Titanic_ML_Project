"""Streamlit dashboard for exploring Titanic survival data."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Page configuration — must be the first Streamlit call in the script
st.set_page_config(
    page_title="Titanic Survival Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color palette used across all charts
COLOR_PRIMARY = "#0B3C5D"      # navy
COLOR_SECONDARY = "#328CC1"    # teal-blue
COLOR_ACCENT = "#D9822B"       # muted coral/orange


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and lightly prepare the Titanic dataset for the dashboard."""
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "train.csv"

    data = pd.read_csv(data_path)

    # Median imputation for Age 
    data["Age"] = data["Age"].fillna(data["Age"].median())

    # Age grouping, consistent with the notebook's bivariate analysis
    bins = [0, 12, 18, 35, 60, 100]
    labels = ["Child (0-12)", "Teen (13-18)", "Young Adult (19-35)",
              "Adult (36-60)", "Senior (60+)"]
    data["AgeGroup"] = pd.cut(data["Age"], bins=bins, labels=labels)

    return data


df = load_data()

# Sidebar
st.sidebar.title("Titanic Dashboard")
st.sidebar.markdown(
    "Explore survival patterns from the 1912 Titanic disaster, filtered by "
    "passenger class."
)

pclass_options = ["All Classes", "1st Class", "2nd Class", "3rd Class"]
pclass_map = {"1st Class": 1, "2nd Class": 2, "3rd Class": 3}

selected_class = st.sidebar.selectbox(
    "Filter by Passenger Class",
    options=pclass_options,
    index=0,
    help="Narrow the dashboard down to a single passenger class.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data source:** Titanic passenger manifest (891 records)\n\n"
    "**Built with:** Streamlit + Plotly"
)

# Apply the filter
if selected_class == "All Classes":
    filtered_df = df.copy()
else:
    filtered_df = df[df["Pclass"] == pclass_map[selected_class]].copy()

# Page header
st.title("Titanic Survival Analysis Dashboard")
st.markdown(
    f"Currently viewing: **{selected_class}** &nbsp;|&nbsp; "
    f"**{filtered_df.shape[0]}** passengers"
)
st.markdown("---")

# KPI row
total_passengers = filtered_df.shape[0]
survival_rate = filtered_df["Survived"].mean() * 100
avg_fare_survived = filtered_df.loc[filtered_df["Survived"] == 1, "Fare"].mean()
avg_fare_not_survived = filtered_df.loc[filtered_df["Survived"] == 0, "Fare"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(label="Total Passengers", value=f"{total_passengers:,}")
kpi2.metric(label="Survival Rate", value=f"{survival_rate:.1f}%")
kpi3.metric(label="Avg Fare (Survived)", value=f"${avg_fare_survived:,.2f}")
kpi4.metric(label="Avg Fare (Did Not Survive)", value=f"${avg_fare_not_survived:,.2f}")

st.markdown("---")

# Chart 1: Survival rate by sex
col1, col2 = st.columns(2)

with col1:
    st.subheader("Survival Rate by Sex")

    sex_survival = (
        filtered_df.groupby("Sex")["Survived"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"Survived": "SurvivalRate"})
    )

    fig_sex = px.bar(
        sex_survival,
        x="Sex",
        y="SurvivalRate",
        color="Sex",
        color_discrete_sequence=[COLOR_ACCENT, COLOR_PRIMARY],
        labels={"SurvivalRate": "Survival Rate (%)", "Sex": "Sex"},
        text_auto=".1f",
    )
    fig_sex.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig_sex, use_container_width=True)

    st.caption("Women survived at much higher rates than men.")

# Chart 2: Survival rate by age group
with col2:
    st.subheader("Survival Rate by Age Group")

    age_survival = (
        filtered_df.groupby("AgeGroup", observed=True)["Survived"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"Survived": "SurvivalRate"})
    )

    fig_age = px.bar(
        age_survival,
        x="AgeGroup",
        y="SurvivalRate",
        color_discrete_sequence=[COLOR_SECONDARY],
        labels={"SurvivalRate": "Survival Rate (%)", "AgeGroup": "Age Group"},
        text_auto=".1f",
    )
    fig_age.update_layout(yaxis_range=[0, 100], xaxis_tickangle=-15)
    st.plotly_chart(fig_age, use_container_width=True)

    st.caption("Children had the best odds of survival.")

st.markdown("---")

# Chart 3: Survival rate by passenger class
st.subheader("Survival Rate by Passenger Class")

class_survival = (
    filtered_df.groupby("Pclass")["Survived"]
    .mean()
    .mul(100)
    .reset_index()
    .rename(columns={"Survived": "SurvivalRate"})
)

fig_class = px.bar(
    class_survival,
    x="Pclass",
    y="SurvivalRate",
    color_discrete_sequence=[COLOR_PRIMARY],
    labels={"SurvivalRate": "Survival Rate (%)", "Pclass": "Passenger Class"},
    text_auto=".1f",
)
fig_class.update_layout(
    yaxis_range=[0, 100], xaxis=dict(tickmode="array", tickvals=[1, 2, 3])
)
st.plotly_chart(fig_class, use_container_width=True)

st.caption("1st class survived at more than double the rate of 3rd class.")

st.markdown("---")
st.caption("Part of an end-to-end ML project: EDA → Dashboard → Model → API → Docker.")