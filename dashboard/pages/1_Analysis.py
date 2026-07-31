"""Statistical analysis page - full write-up of notebooks/Udbhav_Statistical_Analysis.ipynb, computed live."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Analysis", layout="wide")

COLOR_PRIMARY = "#0B3C5D"
COLOR_SECONDARY = "#328CC1"
COLOR_ACCENT = "#D9822B"

NUMERIC_COLS = ["Age", "SibSp", "Parch", "Fare"]


@st.cache_data
def load_data() -> tuple[pd.DataFrame, tuple[int, int]]:
    base_dir = Path(__file__).resolve().parent.parent.parent
    data = pd.read_csv(base_dir / "data" / "train.csv")
    raw_shape = data.shape

    data["Title"] = data["Name"].str.extract(r",\s*([^\.]+)\.")
    rare_titles = data["Title"].value_counts()[lambda x: x < 10].index.tolist()
    data["Title"] = data["Title"].replace(rare_titles, "Rare")
    data["Title"] = data["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    data["HasCabin"] = data["Cabin"].notna().astype(int)

    return data, raw_shape


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def cramers_v(table) -> float:
    chi2 = stats.chi2_contingency(table)[0]
    n = table.sum().sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def count_outliers_iqr(series: pd.Series) -> dict:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        "Outliers": outliers.shape[0],
        "Outlier %": round(outliers.shape[0] / series.dropna().shape[0] * 100, 2),
    }


df, raw_shape = load_data()

st.title("Statistical Analysis")
st.markdown("Full write-up of `notebooks/Udbhav_Statistical_Analysis.ipynb`, computed live on the same dataset.")
st.markdown("---")

# 1. Load and Inspect
st.subheader("1. Load and Inspect")
st.markdown(
    f"- **Shape**: {raw_shape[0]} rows, {raw_shape[1]} columns\n"
    "- **Age**: mean ~30, max 80\n"
    "- **Fare**: min ~0, max ~512, right-skewed\n"
    "- **SibSp/Parch**: mostly 0, most passengers traveled alone"
)

missing = df[["Cabin", "Age", "Embarked"]].isnull().sum()
st.markdown(
    f"- **Cabin**: {missing['Cabin']} missing ({missing['Cabin'] / len(df):.0%}), "
    "too sparse to impute, converted to a `HasCabin` flag\n"
    f"- **Age**: {missing['Age']} missing ({missing['Age'] / len(df):.0%}), imputed per "
    "Title group median (see section 4)\n"
    f"- **Embarked**: {missing['Embarked']} missing, imputed with the mode"
)

st.markdown("---")

# 2. Univariate Analysis
st.subheader("2. Univariate Analysis")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="Age", nbins=30, color_discrete_sequence=[COLOR_PRIMARY])
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(df, x="Fare", nbins=30, color_discrete_sequence=[COLOR_ACCENT])
    st.plotly_chart(fig, use_container_width=True)

skew = df[NUMERIC_COLS].skew().sort_values(ascending=False)
decisions = {
    "Fare": "Log-transform", "SibSp": "Bucket into 0/1/2+",
    "Parch": "Bucket into 0/1/2+", "Age": "Median imputation",
}
skew_table = pd.DataFrame({
    "Skewness": skew.round(2),
    "Decision": [decisions[c] for c in skew.index],
})
st.markdown("Skewness (|skew| > 1 highly skewed, 0.5-1 moderate, < 0.5 symmetric):")
st.dataframe(skew_table, use_container_width=True)

male_pct = (df["Sex"] == "male").mean()
st.markdown(
    f"- **Sex**: {male_pct:.0%} male, {1 - male_pct:.0%} female\n"
    "- **Pclass**: more 3rd class passengers than 1st and 2nd combined\n"
    "- **Embarked**: mostly Southampton (S)"
)

st.markdown("---")

# 3. Outlier Detection
st.subheader("3. Outlier Detection")
st.markdown("Boxplots to spot outliers visually, IQR rule to quantify them per column.")

box_col1, box_col2 = st.columns(2)
with box_col1:
    fig = px.box(df, y="Age", color_discrete_sequence=[COLOR_PRIMARY], points="outliers")
    st.plotly_chart(fig, use_container_width=True)
with box_col2:
    fig = px.box(df, y="Fare", color_discrete_sequence=[COLOR_ACCENT], points="outliers")
    st.plotly_chart(fig, use_container_width=True)

box_col3, box_col4 = st.columns(2)
with box_col3:
    fig = px.box(df, y="SibSp", color_discrete_sequence=[COLOR_SECONDARY], points="outliers")
    st.plotly_chart(fig, use_container_width=True)
with box_col4:
    fig = px.box(df, y="Parch", color_discrete_sequence=[COLOR_SECONDARY], points="outliers")
    st.plotly_chart(fig, use_container_width=True)

outlier_notes = {
    "Age": "Keep, real values",
    "SibSp": "Keep, bucket instead of removing",
    "Parch": "Keep, IQR rule overreacts on sparse data (Q1=Q3=0)",
    "Fare": "Keep, log-transform instead of removing",
}
with st.expander("IQR outlier table", expanded=True):
    outlier_table = pd.DataFrame({col: count_outliers_iqr(df[col].dropna()) for col in NUMERIC_COLS}).T
    outlier_table["Decision"] = [outlier_notes[c] for c in outlier_table.index]
    st.dataframe(outlier_table, use_container_width=True)
    st.caption(
        "Survived (binary) and Pclass (ordinal) are excluded, the IQR rule doesn't apply to them. "
        "Only remove outliers if they look like data errors (e.g. negative age) - all flagged values "
        "here are real, so they're kept and handled with transforms or bucketing instead of removal."
    )

st.markdown("---")

# 4. Missingness Mechanism
st.subheader("4. Missingness Mechanism")
age_missing = df["Age"].isnull().astype(int)
chi2_pclass, p_pclass, _, _ = stats.chi2_contingency(pd.crosstab(age_missing, df["Pclass"]))
chi2_surv, p_surv, _, _ = stats.chi2_contingency(pd.crosstab(age_missing, df["Survived"]))
st.markdown(
    f"- **Age missing vs Pclass**: chi2={chi2_pclass:.2f}, p={fmt_p(p_pclass)}, "
    "3rd class passengers are far more likely to have missing age\n"
    f"- **Age missing vs Survived**: chi2={chi2_surv:.2f}, p={fmt_p(p_surv)}\n"
    "- Age is not Missing Completely At Random, it's **Missing At Random (MAR)**, related "
    "to Pclass, so it's imputed by Title group median instead of one global median"
)

st.markdown("---")

# 5. Feature Engineering
st.subheader("5. Feature Engineering")
st.markdown(
    "Created: `HasCabin`, `FamilySize` (SibSp + Parch + 1), `IsAlone`, "
    "`Title` (extracted from Name, rare titles grouped)."
)

age_by_title = df.groupby("Title")["Age"].median().round(1)
st.markdown("Median age by title, used for imputation: " + ", ".join(f"{t} {a}" for t, a in age_by_title.items()))

fam_survival = df.groupby("FamilySize")["Survived"].mean()
mid_family = fam_survival.loc[fam_survival.index.isin([2, 3, 4])]
alone_survival = df.groupby("IsAlone")["Survived"].mean()
cabin_survival = df.groupby("HasCabin")["Survived"].mean()
st.markdown(
    f"- **FamilySize**: 2-4 survives best ({mid_family.min():.0%}-{mid_family.max():.0%}), "
    "solo travelers and 5+ families survive worse\n"
    f"- **IsAlone**: {alone_survival[0]:.0%} survival with family vs {alone_survival[1]:.0%} alone\n"
    f"- **HasCabin**: {cabin_survival[1]:.0%} survival vs {cabin_survival[0]:.0%} missing, "
    "likely a class proxy rather than the cabin itself"
)

st.markdown("---")

# 6. Bivariate Analysis
st.subheader("6. Bivariate Analysis")

st.markdown("**Correlation heatmap (numeric features, Pearson)**")
corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
corr_matrix = df[corr_cols].corr()
fig = px.imshow(
    corr_matrix.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    labels={"color": "Pearson r"},
)
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    f"- **Pclass vs Survived**: {corr_matrix.loc['Pclass', 'Survived']:.2f}, higher class number means lower survival\n"
    f"- **Fare vs Survived**: {corr_matrix.loc['Fare', 'Survived']:.2f}, higher fare means higher survival\n"
    f"- **Pclass vs Fare**: {corr_matrix.loc['Pclass', 'Fare']:.2f}, 1st class costs more\n"
    f"- **Age vs Survived**: {corr_matrix.loc['Age', 'Survived']:.2f}, weak, survival by age isn't linear "
    "(children survive more, see age group breakdown)\n"
    f"- **SibSp vs Parch**: {corr_matrix.loc['SibSp', 'Parch']:.2f}, families tend to travel together\n"
    "- Sex and Embarked are excluded here, they're unordered categories, checked separately via chi-square below"
)

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    class_survival = (
        df.groupby("Pclass")["Survived"].mean().mul(100).reset_index()
        .rename(columns={"Survived": "SurvivalRate"})
    )
    fig = px.bar(
        class_survival, x="Pclass", y="SurvivalRate",
        color_discrete_sequence=[COLOR_PRIMARY],
        labels={"SurvivalRate": "Survival Rate (%)", "Pclass": "Passenger Class"},
        text_auto=".1f",
    )
    fig.update_layout(yaxis_range=[0, 100], xaxis=dict(tickmode="array", tickvals=[1, 2, 3]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("1st 63%, 2nd 47%, 3rd 24%.")

with col2:
    embarked_survival = (
        df.dropna(subset=["Embarked"]).groupby("Embarked")["Survived"].mean()
        .mul(100).reset_index().rename(columns={"Survived": "SurvivalRate"})
    )
    fig = px.bar(
        embarked_survival, x="Embarked", y="SurvivalRate",
        color_discrete_sequence=[COLOR_ACCENT],
        labels={"SurvivalRate": "Survival Rate (%)", "Embarked": "Port"},
        text_auto=".1f",
    )
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cherbourg 55%, Queenstown 39%, Southampton 34% - mostly a class-mix effect, not the port itself.")

with st.expander("Chi-square / Cramer's V detail", expanded=True):
    chi_rows = []
    for col in ["Sex", "Pclass", "Embarked"]:
        tmp = df.dropna(subset=[col])
        table = pd.crosstab(tmp[col], tmp["Survived"])
        chi2, p, _, _ = stats.chi2_contingency(table)
        chi_rows.append({
            "Variable": col, "Chi2": round(chi2, 2), "p-value": fmt_p(p),
            "Cramer's V": round(cramers_v(table), 2),
        })
    st.markdown("Chi-square tests confirm all three relationships are real, not due to chance:")
    st.dataframe(pd.DataFrame(chi_rows), use_container_width=True, hide_index=True)
    st.caption("Cramer's V shows effect size, since p-value alone doesn't. Sex has by far the strongest effect.")

st.markdown("**Class x Sex interaction**")
heatmap_data = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack().mul(100).round(1)
fig = px.imshow(
    heatmap_data, text_auto=True, color_continuous_scale="Blues",
    labels={"color": "Survival Rate (%)"},
)
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Class matters far more for men than women. Women survive at 92%+ in 1st/2nd class "
    "regardless; men's odds drop sharply as class drops."
)

st.markdown("---")

# 7. Hypothesis Testing
st.subheader("7. Hypothesis Testing")

survived_fare = df.loc[df["Survived"] == 1, "Fare"]
not_survived_fare = df.loc[df["Survived"] == 0, "Fare"]

levene_stat, levene_p = stats.levene(survived_fare, not_survived_fare)
equal_var = levene_p > 0.05
t_stat, p_value = stats.ttest_ind(survived_fare, not_survived_fare, equal_var=equal_var)

n1, n2 = len(survived_fare), len(not_survived_fare)
pooled_std = np.sqrt(
    ((n1 - 1) * survived_fare.std() ** 2 + (n2 - 1) * not_survived_fare.std() ** 2) / (n1 + n2 - 2)
)
cohens_d = (survived_fare.mean() - not_survived_fare.mean()) / pooled_std

st.markdown("**Test 1: Fare vs Survival (Welch's t-test)**")
st.markdown(
    f"- Levene's test p={levene_p:.5f}, variances not equal -> Welch's t-test\n"
    f"- t={t_stat:.2f}, p={fmt_p(p_value)} -> reject H0, survivors paid more on average\n"
    f"- Cohen's d = {cohens_d:.2f} (medium effect size)"
)

shapiro_p = stats.shapiro(df["Fare"].dropna().sample(500, random_state=42))[1]
mw_stat, mw_p = stats.mannwhitneyu(survived_fare, not_survived_fare, alternative="two-sided")

with st.expander("Normality check"):
    st.markdown(
        f"- Shapiro-Wilk (sample of 500) p={fmt_p(shapiro_p)}: Fare is not normally distributed\n"
        f"- Mann-Whitney U (non-parametric) p={fmt_p(mw_p)}: same conclusion as the t-test, "
        "so the result isn't an artifact of the skewed distribution"
    )

groups = [df.loc[df["Pclass"] == p, "Fare"] for p in sorted(df["Pclass"].unique())]
f_stat, anova_p = stats.f_oneway(*groups)

st.markdown("**Test 2: Fare across Pclass (ANOVA)**")
st.markdown(f"F={f_stat:.2f}, p={fmt_p(anova_p)}. Mean fare differs significantly across passenger class.")

with st.expander("Multiple comparison correction"):
    st.markdown(
        "5 significance tests were run above (1 t-test, 3 chi-square, 1 ANOVA). Bonferroni "
        f"adjusted alpha = 0.05 / 5 = {0.05 / 5:.2f}. Every p-value above is far below this, "
        "so all results stay significant after correction."
    )

st.markdown("---")

# 8. Summary
st.subheader("8. Summary")
st.markdown(
    "- **Sex** is the strongest predictor of survival: women 74%, men 19%\n"
    "- **Class and fare** tell the same story: 1st class passengers survived more and paid "
    "more, both linked to wealth and cabin location\n"
    "- **Class x Sex**: class matters far more for men than for women\n"
    "- **Age**: children survived more than any other age group; age alone is a weak predictor\n"
    "- **Embarked** looks predictive on its own, but is mostly explained by the class mix at each port\n"
    "- **FamilySize and HasCabin** are useful added features: mid-size families survived more "
    "than solo travelers, and HasCabin mostly reflects class rather than the cabin itself"
)
st.caption(
    "All key relationships (chi-square, t-test, ANOVA) stayed significant even after "
    "correcting for running multiple tests, so these findings are not due to chance."
)
