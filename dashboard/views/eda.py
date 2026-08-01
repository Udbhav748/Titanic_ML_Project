"""Data Analysis — the highest-signal findings, interactive."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import load_transformed
from theme import (
    CARD_BORDER,
    ERROR,
    PRIMARY,
    QUATERNARY,
    SUCCESS,
    TERTIARY,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WARNING,
    highlight_card,
    page_header,
    style_fig,
    takeaway,
    what_this_means,
)

BAR_PALETTE = [PRIMARY, SUCCESS, WARNING, QUATERNARY, TERTIARY, ERROR]

df = load_transformed()

page_header("Data Analysis", "The key patterns that shaped which features power the model.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Feature Distributions", "Survival Analysis", "Correlation Heatmap",
    "Class x Sex Interaction", "VIF Analysis", "Feature Selection Journey",
])

with tab1:
    metric = st.radio("Distribution:", ["Age", "Fare"], horizontal=True)
    fig = px.histogram(
        df, x=metric, color="Survived", barmode="overlay", opacity=0.65,
        color_discrete_map={0: ERROR, 1: PRIMARY},
        labels={"Survived": "Outcome"},
    )
    fig.for_each_trace(lambda t: t.update(name="Survived" if t.name == "1" else "Did Not Survive"))
    with st.container(border=True, key="card-feature-dist-chart"):
        st.plotly_chart(style_fig(fig), use_container_width=True)
    if metric == "Fare":
        takeaway("Fare tells us more about survival than Age does, but it's heavily skewed, so we applied a log transform first.")
        what_this_means(
            observation="Most passengers paid a low fare, but a small group paid a lot more, "
            "giving the chart a long tail.",
            impact="Skewed data like this can throw off a linear model, since it works best "
            "when values are spread out more evenly.",
            decision="We applied a log transform to Fare before feeding it into the model.",
        )
    else:
        takeaway("Age looks almost the same for both groups, except young children survived a bit more often.")
        what_this_means(
            observation="People who survived and people who didn't show very similar age "
            "patterns, apart from a small bump in survival among young children.",
            impact="Age on its own does not split the two outcomes very well.",
        )

with tab2:
    filter_col, _ = st.columns([1, 3])
    with filter_col:
        factor = st.selectbox("Break down survival rate by:", ["Sex", "Pclass", "Title", "Embarked"])
    rate = df.groupby(factor, observed=True)["Survived"].mean().sort_values(ascending=False).mul(100)
    overall_rate = df["Survived"].mean() * 100

    with st.container(border=True, key="card-survival-chart"):
        _, chart_col, _ = st.columns([1, 10, 1])
        with chart_col:
            fig = go.Figure()
            bar_colors = [BAR_PALETTE[i % len(BAR_PALETTE)] for i in range(len(rate))]
            fig.add_trace(go.Bar(
                x=rate.index.astype(str), y=rate.values,
                marker_color=bar_colors, marker_cornerradius=8,
                text=[f"{v:.0f}%" for v in rate.values],
                textposition="outside", textfont=dict(size=14, color=TEXT_PRIMARY),
                hovertemplate="<b>%{x}</b><br>Survival rate: %{y:.1f}%<extra></extra>",
            ))
            fig.add_hline(
                y=overall_rate, line_dash="dash", line_color=TEXT_MUTED, line_width=1.5,
                annotation_text=f"Overall average: {overall_rate:.1f}%",
                annotation_position="top right", annotation_font=dict(size=12, color=TEXT_MUTED),
            )
            fig = style_fig(fig, height=340)
            fig.update_layout(
                hoverlabel=dict(bgcolor="white", font_size=13, bordercolor=CARD_BORDER),
                bargap=0.4,
            )
            fig.update_xaxes(title=factor, showline=True, linecolor=CARD_BORDER, ticks="", tickfont=dict(size=13))
            fig.update_yaxes(
                showticklabels=False, title="Survival Rate (%)", showline=True, linecolor=CARD_BORDER,
                showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                range=[0, max(rate.values.max(), overall_rate) * 1.25],
            )
            st.plotly_chart(fig, use_container_width=True)

    takeaway("Sex, Passenger Class, and Title are the three strongest predictors, and each one splits survival outcomes clearly.")
    SURVIVAL_WTM = {
        "Sex": dict(
            observation="Women survived at a much higher rate than men.",
            impact="This is one of the clearest patterns in the entire dataset.",
            decision="Sex was kept as one of the core features in the final model.",
        ),
        "Pclass": dict(
            observation="Survival drops sharply as you go from 1st class to 3rd class.",
            impact="Ticket class works as a stand in for wealth and where a passenger's cabin "
            "sat on the ship.",
            decision="Passenger Class stayed in as a core feature.",
        ),
        "Title": dict(
            observation="Passengers with the title Mrs or Miss survived far more often than "
            "those with Mr.",
            impact="Title bundles sex, age, and social status into one field.",
            decision="We engineered Title from the passenger's name and kept it in the final "
            "feature set.",
        ),
        "Embarked": dict(
            observation="Survival rate does shift depending on which port a passenger boarded "
            "at, but not as much as with Sex or Class.",
            impact="It's a real pattern, just a smaller one than the other features.",
        ),
    }
    what_this_means(**SURVIVAL_WTM[factor])

with tab3:
    corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "FarePerPerson", "HasCabin", "IsAlone"]
    corr = df[corr_cols].corr()
    fig = px.imshow(
        corr.round(2), text_auto=True, color_continuous_scale=[ERROR, "#F3F4F6", PRIMARY],
        zmin=-1, zmax=1, labels={"color": "Pearson r"},
    )
    with st.container(border=True, key="card-correlation-chart"):
        st.plotly_chart(style_fig(fig, height=480), use_container_width=True)
    takeaway("FamilySize overlaps heavily with SibSp and Parch, and FarePerPerson tracks Fare almost exactly. The VIF check confirms just how much.")
    what_this_means(
        observation="FamilySize moves closely with SibSp and Parch, and FarePerPerson tracks "
        "Fare almost exactly.",
        impact="When features move together this closely, they are mostly repeating the same "
        "information.",
        decision="We looked at this more closely with a VIF check, a way of measuring how much "
        "features overlap, before deciding what to drop.",
    )

with tab4:
    heatmap_data = df.groupby(["Pclass", "Sex"], observed=True)["Survived"].mean().unstack().mul(100).round(1)
    fig = px.imshow(
        heatmap_data, text_auto=True, color_continuous_scale=["#F3F4F6", PRIMARY],
        labels={"color": "Survival rate (%)"},
    )
    with st.container(border=True, key="card-class-sex-chart"):
        st.plotly_chart(style_fig(fig, height=350), use_container_width=True)
    takeaway("Sex sets the baseline survival rate, and class shifts it further. First class women survived 97% of the time, first class men only 37%.")
    what_this_means(
        observation="Class changes how much Sex matters, since 1st class women survived 97% "
        "of the time while 1st class men only survived 37% of the time.",
        impact="Sex and Class tell a fuller story together than either one does on its own.",
        decision="That's part of why we kept Sex and Pclass as two separate features instead "
        "of merging them.",
    )

with tab5:
    X = df[["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "FarePerPerson", "HasCabin", "IsAlone"]].copy()
    X["Sex_male"] = (df["Sex"] == "male").astype(int)
    X_const = X.assign(const=1.0)
    vif = pd.Series(
        [variance_inflation_factor(X_const.values, i) for i in range(X.shape[1])],
        index=X.columns,
    ).sort_values(ascending=False)
    display_vif = vif.replace(np.inf, 50)
    colors = [ERROR if v > 5 else PRIMARY for v in vif]
    fig = px.bar(
        x=display_vif.values, y=display_vif.index, orientation="h",
        labels={"x": "VIF (capped at 50 for display)", "y": ""},
        text=["∞" if np.isinf(v) else f"{v:.1f}" for v in vif],
    )
    fig.update_traces(marker_color=colors, textposition="outside")
    fig.add_vline(x=5, line_dash="dash", line_color=TEXT_MUTED)
    with st.container(border=True, key="card-vif-chart"):
        st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
    takeaway("SibSp, Parch, and FamilySize overlap completely, so using all three together in a model is not a good idea.")
    what_this_means(
        observation="SibSp, Parch, and FamilySize came back with an infinite VIF score, meaning "
        "they overlap completely.",
        impact="Putting all three into a linear model would make its coefficients unreliable.",
        decision="We kept FamilySize and dropped SibSp and Parch from the final feature set.",
    )

with tab6:
    FSJ_STAGES = [
        ("Candidate Features", "11 Initial Features", "blue"),
        ("Statistical Testing", "Validated predictive significance", "blue"),
        ("Mutual Information", "Measured feature relevance", "purple"),
        ("Multicollinearity Analysis", "Removed redundant features", "purple"),
        ("Engineering Review", "Balanced simplicity and performance", "green"),
        ("Final Feature Set", "8 Selected Features", "green"),
    ]
    FSJ_COLORS = {"blue": PRIMARY, "purple": QUATERNARY, "green": SUCCESS}

    header_col, badge_col = st.columns([3, 1])
    with header_col:
        st.caption("How the final feature set was selected.")
    with badge_col:
        st.markdown(
            f"<div style='text-align:right;'><span style='display:inline-block; background:#EFF6FF; "
            f"border:1px solid {PRIMARY}; border-radius:999px; padding:0.25rem 0.85rem; "
            f"font-weight:700; color:{PRIMARY}; font-size:0.8rem;'>Feature Reduction&nbsp;&nbsp;11 &rarr; 8</span></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    step_weights = []
    for i in range(len(FSJ_STAGES)):
        step_weights.append(6)
        if i < len(FSJ_STAGES) - 1:
            step_weights.append(1)
    step_cols = st.columns(step_weights)
    for i, (title, subtitle, color_key) in enumerate(FSJ_STAGES):
        accent = FSJ_COLORS[color_key]
        with step_cols[i * 2]:
            st.markdown(
                "<div style='text-align:center;'>"
                f"<div style='width:36px; height:36px; border-radius:50%; background:{accent}1A; "
                f"border:2px solid {accent}; color:{accent}; font-weight:700; font-size:0.95rem; "
                "display:flex; align-items:center; justify-content:center; margin:0 auto 0.5rem auto;'>"
                f"{i + 1}</div>"
                f"<div style='font-weight:700; color:{TEXT_PRIMARY}; font-size:0.85rem;'>{title}</div>"
                f"<div style='color:{TEXT_MUTED}; font-size:0.74rem; margin-top:0.15rem;'>{subtitle}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        if i < len(FSJ_STAGES) - 1:
            with step_cols[i * 2 + 1]:
                st.markdown(
                    "<div style='text-align:center; padding-top:10px; color:#CBD5E1; font-size:1.1rem;'>&rarr;</div>",
                    unsafe_allow_html=True,
                )

    st.write("")
    ins_cols = st.columns(3)
    with ins_cols[0]:
        highlight_card(
            "", "Redundancy Removed",
            "Three highly correlated features were removed after multicollinearity analysis.",
            show_icon=False,
        )
    with ins_cols[1]:
        highlight_card(
            "", "Evidence-Based Selection",
            "Every feature was evaluated using statistical testing, information gain, and engineering review.",
            show_icon=False,
        )
    with ins_cols[2]:
        highlight_card(
            "", "Final Outcome",
            "A simpler feature set improved generalization while reducing unnecessary complexity.",
            show_icon=False,
        )
    takeaway("Each feature was kept or removed based on the analysis, not assumptions.")
    what_this_means(
        observation="Statistical tests, mutual information, and the VIF check narrowed the "
        "feature list from 11 down to 8.",
        impact="Each feature we cut was either repeating information or barely helping the "
        "model, so it wasn't worth the extra complexity.",
        decision="Those 8 features are what the model is trained on and what powers every "
        "prediction in this dashboard.",
    )
