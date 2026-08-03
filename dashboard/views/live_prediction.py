"""Live Prediction — a decision support view built entirely from the trained model."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import (
    get_coefficients,
    get_logit_strength_cutoffs,
    get_population_reference,
    get_shap_explainer,
    get_transformed_train_matrix,
    load_model_v2,
    load_train_reference,
)
from theme import (
    CARD_BORDER,
    ERROR,
    PRIMARY,
    QUATERNARY,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WARNING,
    page_header,
    probability_bar,
    style_fig,
)

model = load_model_v2()

PCLASS_LABEL = {1: "1st", 2: "2nd", 3: "3rd"}
EMBARK_LABEL = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}

page_header("Live Prediction", "Enter passenger details and get a real prediction from the trained model.")

st.markdown(
    f"""
    <style>
        div[data-testid="stForm"] {{
            border-top: 5px solid {PRIMARY} !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 4px rgba(17, 24, 39, 0.06);
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<span style='color:{PRIMARY}; font-weight:700; font-size:0.8rem;'>PASSENGER PROFILE</span>", unsafe_allow_html=True)
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=0)
        sex = st.selectbox("Sex", ["female", "male"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0)
    with col2:
        st.markdown(f"<span style='color:{QUATERNARY}; font-weight:700; font-size:0.8rem;'>FAMILY</span>", unsafe_allow_html=True)
        sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, value=0)
        parch = st.number_input("Parents / Children Aboard", min_value=0, value=0)
        fare = st.number_input("Fare Paid ($)", min_value=0.0, value=32.0)
    with col3:
        st.markdown(f"<span style='color:{SUCCESS}; font-weight:700; font-size:0.8rem;'>TRAVEL DETAILS</span>", unsafe_allow_html=True)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
        has_cabin = st.checkbox("Cabin Recorded")
        title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

    submitted = st.form_submit_button("Predict Survival", type="primary", use_container_width=True)


def describe_feature(name: str) -> str:
    """Translate a transformed feature name into a plain-language label for this passenger."""
    if name == "Fare":
        return "High Fare" if fare > pop["fare_median"] else "Low Fare"
    if name == "Age":
        return "Older Age" if age > pop["age_median"] else "Younger Age"
    if name == "FamilySize":
        if family_size == 1:
            return "Traveling Alone"
        return "Larger Family" if family_size > pop["family_median"] else "Smaller Family"
    if name.startswith("Sex_"):
        return f"Sex ({sex.capitalize()})"
    if name.startswith("Pclass_"):
        return f"Passenger Class ({PCLASS_LABEL[pclass]})"
    if name.startswith("Title_"):
        return f"Title ({title})"
    if name.startswith("Embarked_"):
        return f"Port of Embarkation ({embarked})"
    if name == "HasCabin":
        return "Cabin Recorded"
    return name


def evaluate_variant(base_df: pd.DataFrame, column: str, new_value) -> tuple:
    variant = base_df.copy()
    variant[column] = new_value
    proba = model.predict_proba(variant)[0][1]
    pred = model.predict(variant)[0]
    return proba, pred


if submitted:
    pop = get_population_reference()
    family_size = sibsp + parch + 1
    input_df = pd.DataFrame([{
        "Pclass": pclass, "Sex": sex, "Age": age, "Fare": fare,
        "Embarked": embarked, "HasCabin": int(has_cabin),
        "FamilySize": family_size, "Title": title,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    logit = model.decision_function(input_df)[0]

    confidence_gap = abs(probability - 0.5)
    if confidence_gap >= 0.35:
        confidence_label, confidence_color = "High", SUCCESS
    elif confidence_gap >= 0.15:
        confidence_label, confidence_color = "Moderate", WARNING
    else:
        confidence_label, confidence_color = "Low", ERROR

    weak_cut, strong_cut = get_logit_strength_cutoffs()
    abs_logit = abs(logit)
    if abs_logit >= strong_cut:
        strength_label = "Strong"
    elif abs_logit >= weak_cut:
        strength_label = "Moderate"
    else:
        strength_label = "Weak"

    # --- Feature contributions (exact, from the model's own coefficients) ---
    transformed = model.named_steps["preprocessor"].transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed).ravel()
    coef_df = get_coefficients(model)
    contrib = pd.DataFrame({
        "feature": coef_df["feature"], "value": transformed, "contribution": transformed * coef_df["coef"].values,
    })
    active = contrib[contrib["value"].abs() > 1e-9].copy()
    active["abs_contrib"] = active["contribution"].abs()
    active = active.sort_values("abs_contrib", ascending=False).reset_index(drop=True)
    total_abs = active["abs_contrib"].sum()
    active["relative_pct"] = active["abs_contrib"] / total_abs * 100 if total_abs > 0 else 0.0
    n_active = len(active)
    strength_bounds = [n_active / 3, 2 * n_active / 3]

    def contrib_strength(rank: int) -> str:
        if rank < strength_bounds[0]:
            return "Strong"
        elif rank < strength_bounds[1]:
            return "Moderate"
        return "Weak"

    active["strength"] = [contrib_strength(i) for i in range(n_active)]
    active["label"] = [describe_feature(f) for f in active["feature"]]
    positive = active[active["contribution"] > 0].sort_values("abs_contrib", ascending=False)
    negative = active[active["contribution"] < 0].sort_values("abs_contrib", ascending=False)

    # --- SHAP attribution — same underlying linear model, computed via
    # shap.LinearExplainer instead of a raw coefficient x value product.
    # For a linear model these are exact (they sum precisely to
    # decision_function), so this is a cross-check on `active` above as
    # much as it is a second visualization. ---
    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(transformed.reshape(1, -1))[0]
    shap_base = float(explainer.expected_value)
    shap_df = pd.DataFrame({"feature": coef_df["feature"], "shap_value": shap_values})
    shap_active = shap_df[shap_df["feature"].isin(active["feature"])].copy()
    shap_active["label"] = [describe_feature(f) for f in shap_active["feature"]]
    shap_active = shap_active.reindex(
        shap_active["shap_value"].abs().sort_values(ascending=False).index
    )

    # --- Counterfactual candidates — realistic values only, grounded in the real dataset ---
    candidates = []
    for pc in (1, 2, 3):
        if pc != pclass:
            candidates.append({"feature": "Passenger Class", "column": "Pclass",
                                "original": f"{PCLASS_LABEL[pclass]} Class", "modified": f"{PCLASS_LABEL[pc]} Class",
                                "new_value": pc})
    other_sex = "male" if sex == "female" else "female"
    candidates.append({"feature": "Sex", "column": "Sex", "original": sex.capitalize(),
                        "modified": other_sex.capitalize(), "new_value": other_sex})
    for e in ("S", "C", "Q"):
        if e != embarked:
            candidates.append({"feature": "Embarked", "column": "Embarked",
                                "original": EMBARK_LABEL[embarked], "modified": EMBARK_LABEL[e], "new_value": e})
    for val in sorted({round(pop["age_q25"]), round(pop["age_median"]), round(pop["age_q75"])}):
        if abs(val - age) >= 2:
            candidates.append({"feature": "Age", "column": "Age",
                                "original": f"{age:.0f}", "modified": f"{val:.0f}", "new_value": float(val)})
    for val in sorted({round(pop["fare_q25"], 2), round(pop["fare_median"], 2), round(pop["fare_q75"], 2)}):
        if abs(val - fare) >= 1:
            candidates.append({"feature": "Fare", "column": "Fare",
                                "original": f"${fare:.2f}", "modified": f"${val:.2f}", "new_value": float(val)})
    for delta in (-1, 1):
        new_fs = family_size + delta
        if 1 <= new_fs <= pop["family_max"] and new_fs != family_size:
            candidates.append({"feature": "Family Size", "column": "FamilySize",
                                "original": str(family_size), "modified": str(new_fs), "new_value": new_fs})

    cf_rows = []
    for c in candidates:
        new_proba, new_pred = evaluate_variant(input_df, c["column"], c["new_value"])
        cf_rows.append({
            "Feature": c["feature"], "Original Value": c["original"], "Modified Value": c["modified"],
            "Updated Probability": new_proba, "Probability Difference": new_proba - probability,
            "Flips Prediction": int(new_pred) != int(prediction),
        })
    cf_df = pd.DataFrame(cf_rows).sort_values(
        "Probability Difference", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)

    flips = cf_df[cf_df["Flips Prediction"]]
    sensitivity = cf_df.groupby("Feature")["Probability Difference"].apply(lambda s: s.abs().max()).sort_values(ascending=False)

    # ============================================================
    # 1. Prediction Summary
    # ============================================================
    st.divider()
    st.subheader("Prediction Summary")
    with st.container(border=True, key="card-prediction-summary"):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Prediction</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:1.3rem; font-weight:700; color:{TEXT_PRIMARY};'>"
                f"{'Survived' if prediction == 1 else 'Did Not Survive'}</div>", unsafe_allow_html=True,
            )
        with s2:
            st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Probability</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.3rem; font-weight:700; color:{TEXT_PRIMARY};'>{probability:.1%}</div>", unsafe_allow_html=True)
        with s3:
            st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Confidence</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.3rem; font-weight:700; color:{confidence_color};'>{confidence_label}</div>", unsafe_allow_html=True)
        with s4:
            st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>Decision Strength</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.3rem; font-weight:700; color:{TEXT_PRIMARY};'>{strength_label}</div>", unsafe_allow_html=True)

    # ============================================================
    # 2. Probability
    # ============================================================
    st.write("")
    st.subheader("Probability")
    probability_bar(probability, confidence_color)
    st.caption(f"{probability:.1%} survival probability — {confidence_label} Confidence")

    # ============================================================
    # 3. Feature Contribution Analysis
    # ============================================================
    st.write("")
    st.subheader("Feature Contribution Analysis")
    pos_col, neg_col = st.columns(2)
    with pos_col:
        with st.container(border=True, key="card-positive-contrib"):
            st.markdown("**Positive Contributors**")
            if positive.empty:
                st.caption("No features push this prediction toward survival.")
            for _, r in positive.iterrows():
                st.markdown(f"- {r['label']} — {r['relative_pct']:.0f}% ({r['strength']})")
    with neg_col:
        with st.container(border=True, key="card-negative-contrib"):
            st.markdown("**Negative Contributors**")
            if negative.empty:
                st.caption("No features push this prediction away from survival.")
            for _, r in negative.iterrows():
                st.markdown(f"- {r['label']} — {r['relative_pct']:.0f}% ({r['strength']})")

    # ============================================================
    # 4. SHAP Feature Attribution
    # ============================================================
    st.write("")
    st.subheader("SHAP Feature Attribution")
    st.caption("This prediction decomposed with SHAP, the industry-standard method for explaining individual predictions.")
    # Track the running cumulative total at every step (not just each bar's
    # own delta) so the y-axis range can be padded around the waterfall's
    # actual highs and lows — outside text labels get clipped otherwise.
    running = shap_base
    running_values = [running]
    for v in shap_active["shap_value"]:
        running += v
        running_values.append(running)
    running_values.append(logit)
    y_max, y_min = max(running_values), min(running_values)
    pad = (y_max - y_min) * 0.15 or 0.1

    waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(shap_active) + ["total"],
        x=["Base Rate"] + shap_active["label"].tolist() + ["This Prediction"],
        y=[shap_base] + shap_active["shap_value"].tolist() + [0],
        connector=dict(line=dict(color=CARD_BORDER)),
        increasing=dict(marker=dict(color=PRIMARY)),
        decreasing=dict(marker=dict(color=ERROR)),
        totals=dict(marker=dict(color=SUCCESS)),
        text=[f"{shap_base:+.2f}"] + [f"{v:+.2f}" for v in shap_active["shap_value"]] + [f"{logit:.2f}"],
        textposition="outside",
    ))
    waterfall.update_layout(yaxis_title="Log-odds of survival", showlegend=False)
    waterfall.update_yaxes(range=[y_min - pad, y_max + pad])
    st.plotly_chart(style_fig(waterfall, height=420), use_container_width=True)
    st.caption(
        f"Base rate {shap_base:+.2f} log-odds, ending at {logit:.2f} log-odds for this "
        f"passenger — a predicted survival probability of {probability:.1%}."
    )

    # ============================================================
    # 5. Counterfactual Decision Explorer
    # ============================================================
    st.write("")
    st.subheader("Counterfactual Decision Explorer")
    display_cf = cf_df[["Feature", "Original Value", "Modified Value", "Updated Probability", "Probability Difference"]].copy()
    display_cf["Updated Probability"] = cf_df["Updated Probability"].map(lambda v: f"{v:.1%}")
    display_cf["Probability Difference"] = cf_df["Probability Difference"].map(lambda v: f"{v:+.1%}")
    st.dataframe(display_cf, use_container_width=True, hide_index=True)

    if not flips.empty:
        best_flip = flips.loc[flips["Probability Difference"].abs().idxmin()]
        st.markdown(
            f"**Smallest change that reverses this prediction:** {best_flip['Feature']} from "
            f"{best_flip['Original Value']} to {best_flip['Modified Value']} — probability moves from "
            f"{probability:.1%} to {best_flip['Updated Probability']:.1%}."
        )
    else:
        top = cf_df.iloc[0]
        st.markdown(
            f"**No realistic single-feature change reverses this prediction.** The most impactful tested "
            f"change is {top['Feature']} from {top['Original Value']} to {top['Modified Value']} "
            f"({top['Probability Difference']:+.1%})."
        )

    # ============================================================
    # 6. Prediction Sensitivity
    # ============================================================
    st.write("")
    st.subheader("Prediction Sensitivity")
    fig = px.bar(
        x=sensitivity.values, y=sensitivity.index, orientation="h",
        labels={"x": "Maximum Probability Impact", "y": ""},
    )
    fig.update_traces(marker_color=PRIMARY)
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, height=320), use_container_width=True)
    st.caption("Impact is the largest probability change observed for each feature during counterfactual analysis.")

    # ============================================================
    # 7. Similar Historical Passengers
    # ============================================================
    st.write("")
    st.subheader("Similar Historical Passengers")
    X_train, y_train = load_train_reference()
    train_matrix = get_transformed_train_matrix()
    distances = np.linalg.norm(train_matrix - transformed, axis=1)
    similarity_pct = 100 / (1 + distances)
    top3 = np.argsort(distances)[:3]

    sim_cols = st.columns(3)
    for col, i in zip(sim_cols, top3):
        idx = X_train.index[i]
        row = X_train.loc[idx]
        survived = y_train.loc[idx] == 1
        with col:
            with st.container(border=True, key=f"card-similar-{idx}"):
                st.markdown(f"**{similarity_pct[i]:.0f}% Similar**")
                st.caption(
                    f"{row['Sex'].capitalize()}, Age {row['Age']:.0f}, {PCLASS_LABEL[row['Pclass']]} Class, "
                    f"Title: {row['Title']}"
                )
                color = SUCCESS if survived else ERROR
                st.markdown(
                    f"<span style='color:{color}; font-weight:700;'>"
                    f"{'Survived' if survived else 'Did Not Survive'}</span>",
                    unsafe_allow_html=True,
                )

    # ============================================================
    # 8. Prediction Stability
    # ============================================================
    st.write("")
    st.subheader("Prediction Stability")
    max_swing = cf_df["Probability Difference"].abs().max()
    if not flips.empty:
        stability_label = "Unstable"
    elif max_swing >= 0.15:
        stability_label = "Moderately Stable"
    elif max_swing >= 0.05:
        stability_label = "Stable"
    else:
        stability_label = "Very Stable"
    top_feature, bottom_feature = sensitivity.index[0], sensitivity.index[-1]
    st.markdown(f"**{stability_label}**")
    st.caption(
        f"{top_feature} has the greatest influence on this prediction (up to {sensitivity.iloc[0]:.1%} probability "
        f"shift); {bottom_feature} has the least (up to {sensitivity.iloc[-1]:.1%})."
    )

    # ============================================================
    # 9. Decision Summary
    # ============================================================
    st.write("")
    st.subheader("Decision Summary")
    summary_lines = [
        f"This passenger is predicted to {'survive' if prediction == 1 else 'not survive'} with "
        f"{confidence_label.lower()} confidence ({probability:.1%}).",
    ]
    if not positive.empty:
        summary_lines.append(f"{positive.iloc[0]['label']} contributes most positively toward survival.")
    if not negative.empty:
        summary_lines.append(f"{negative.iloc[0]['label']} decreases the survival probability the most.")
    summary_lines.append(f"Changing {top_feature} would have the greatest impact on this prediction.")
    st.markdown("\n".join(f"- {line}" for line in summary_lines[:4]))
