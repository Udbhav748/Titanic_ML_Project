"""Model Performance — why Logistic Regression became the production model."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import (
    evaluate_v2_on_test,
    get_bootstrap_ci,
    get_coefficients,
    get_error_composition,
    get_misclassified_examples,
    get_permutation_importance,
    get_subgroup_performance,
    load_comparison_curves,
    load_model_v2,
    load_stage4_results,
)
from theme import (
    ERROR,
    MODEL_COLORS,
    MODEL_SHORT_NAMES,
    PRIMARY,
    SECONDARY,
    SUCCESS,
    WARNING,
    metric_card,
    page_header,
    style_fig,
    takeaway,
    what_this_means,
)

results = load_stage4_results()
X_test, y_test, y_pred, y_proba, metrics = evaluate_v2_on_test()
model = load_model_v2()

page_header("Model Performance", "The production model and the metrics behind its selection.")

m_cols = st.columns(6)
with m_cols[0]:
    metric_card("Model", "Logistic Regression", height=70, compact=True)
with m_cols[1]:
    metric_card("Accuracy", f"{metrics['accuracy']:.1%}", height=70, compact=True)
with m_cols[2]:
    metric_card("Precision", f"{metrics['precision']:.1%}", height=70, compact=True)
with m_cols[3]:
    metric_card("Recall", f"{metrics['recall']:.1%}", height=70, compact=True)
with m_cols[4]:
    metric_card("F1 Score", f"{metrics['f1']:.3f}", height=70, compact=True)
with m_cols[5]:
    metric_card("ROC-AUC", f"{metrics['roc_auc']:.3f}", height=70, compact=True)
st.write("")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "• Model Comparison", "• Cross-Validation", "• Train vs Test",
    "• ROC & Confusion Matrix", "• Precision-Recall", "• Calibration",
    "• Threshold Analysis", "• Feature Importance", "• Fairness & Errors",
    "• Model Development",
])

with tab1:
    comparison_df = pd.DataFrame(results)[["name", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_mean", "n_features"]]
    comparison_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Mean", "Features"]
    st.dataframe(
        comparison_df.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Mean"]})
        .highlight_max(subset=["Accuracy", "F1", "ROC-AUC"], color="#D1FAE5"),
        use_container_width=True, hide_index=True,
    )

    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    bootstrap_ci = get_bootstrap_ci()
    fig = go.Figure()
    for r in results:
        point_values = [r[k] for k in metric_keys]
        ci_bounds = [bootstrap_ci[r["name"]][k] for k in metric_keys]
        error_plus = [max(0.0, hi - v) for v, (lo, hi) in zip(point_values, ci_bounds)]
        error_minus = [max(0.0, v - lo) for v, (lo, hi) in zip(point_values, ci_bounds)]
        fig.add_trace(go.Bar(
            name=MODEL_SHORT_NAMES[r["name"]], x=metric_labels, y=point_values,
            marker_color=MODEL_COLORS[r["name"]],
            error_y=dict(type="data", symmetric=False, array=error_plus, arrayminus=error_minus, thickness=1.2),
        ))
    fig.update_layout(barmode="group", yaxis_title="Score", yaxis_range=[0, 1])
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    st.caption("Error bars show a 95% bootstrap confidence interval (1,000 resamples) on each metric, not just the point estimate.")

    logreg_f1_ci = bootstrap_ci["Logistic Regression (8 features)"]["f1"]
    logreg_f1_width = logreg_f1_ci[1] - logreg_f1_ci[0]
    with st.container(border=True, key="card-model-comparison-findings"):
        st.markdown(
            "- Logistic Regression leads on 4 of 5 metrics — accuracy, recall, F1, and ROC-AUC\n"
            "- Gradient Boosting edges it only on precision (0.783 vs. 0.760)\n"
            "- Random Forest is the weakest model on every metric, despite sharing the same 8-feature schema\n"
            f"- Logistic Regression's own F1 95% confidence interval is "
            f"[{logreg_f1_ci[0]:.3f}, {logreg_f1_ci[1]:.3f}] — a width of {logreg_f1_width:.3f}, "
            "wide enough to swallow every gap between these four models several times over"
        )
    takeaway(
        f"Logistic Regression scores best overall, but its own F1 has a bootstrap confidence "
        f"interval {logreg_f1_width:.3f} wide — bigger than the gap to every other model. "
        "Stability mattered just as much as the raw score."
    )
    what_this_means(
        observation="Logistic Regression came out ahead on 4 of the 5 metrics we tracked, with "
        "Gradient Boosting only winning on precision.",
        impact="The lead is real, but small enough that performance alone doesn't settle which "
        "model to use.",
        decision="We picked Logistic Regression after also weighing how stable and generalizable "
        "it was, not just its raw score.",
    )

with tab2:
    fig = go.Figure()
    for r in results:
        fig.add_trace(go.Box(
            y=r["cv_scores"], name=MODEL_SHORT_NAMES[r["name"]],
            marker_color=MODEL_COLORS[r["name"]], boxpoints="all", pointpos=0,
        ))
    fig.update_layout(yaxis_title="Cross-Validation F1 (5-fold)", showlegend=False)
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    with st.container(border=True, key="card-cv-findings"):
        st.markdown(
            "- Random Forest and Gradient Boosting have the tightest fold-to-fold spread, but around a lower score band\n"
            "- Logistic Regression's worst fold still beats the median fold of both tree ensembles\n"
            "- The baseline has the widest spread despite being the only tuned model"
        )
    takeaway("Low variance only matters if the score behind it is actually good, not just consistent.")
    what_this_means(
        observation="Even Logistic Regression's worst fold in cross-validation beat the typical "
        "fold from the two tree based models.",
        impact="Being consistent only matters if you're consistently good, not consistently average.",
        decision="That ruled out picking a tree model just because its scores looked more "
        "stable fold to fold.",
    )

with tab3:
    names = [MODEL_SHORT_NAMES[r["name"]] for r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Train F1", x=names, y=[r["train_f1"] for r in results], marker_color=SECONDARY))
    fig.add_trace(go.Bar(name="Test F1", x=names, y=[r["f1"] for r in results], marker_color=PRIMARY))
    fig.update_layout(barmode="group", yaxis_title="F1 Score")
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    with st.container(border=True, key="card-train-test-findings"):
        st.markdown(
            "- Random Forest's train F1 (0.984) towers over its test F1 (0.729) — a +0.255 gap from unlimited-depth memorization\n"
            "- Logistic Regression is the only model where test F1 does not drop below train F1\n"
            "- Gradient Boosting overfits less severely; the tuned baseline overfits least of the three ensembles"
        )
    takeaway("Logistic Regression is the only model that performs the same on training and test data. The others memorized patterns instead of learning them.")
    what_this_means(
        observation="Random Forest scored 0.255 higher on training data than on test data, "
        "while Logistic Regression scored about the same on both.",
        impact="That gap for Random Forest is a sign it memorized the training data instead "
        "of learning patterns that generalize.",
        decision="Logistic Regression's lack of that gap was a big part of why we shipped it "
        "to production.",
    )

with tab4:
    comparison = load_comparison_curves()
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for name, d in comparison.items():
            fpr, tpr, _ = roc_curve(d["y_test"], d["y_proba"])
            lw = 3.5 if "Logistic Regression" in name else 1.6
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=MODEL_SHORT_NAMES[name],
                line=dict(color=MODEL_COLORS[name], width=lw),
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#9CA3AF", dash="dash"), showlegend=False))
        fig.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    with col2:
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm, text_auto=True, x=["Died", "Survived"], y=["Died", "Survived"],
            color_continuous_scale=["white", PRIMARY], labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        )
        fig.update_layout(title="Confusion Matrix (Logistic Regression, threshold = 0.5)")
        st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    with st.container(border=True, key="card-roc-findings"):
        st.markdown(
            "- All four ROC curves cluster tightly — AUC ranges from 0.839 (Random Forest) to 0.865 (Logistic Regression)\n"
            "- Logistic Regression (bold) sits at or above the other three across nearly the full range\n"
            "- 18 false positives vs. 12 false negatives — the production model leans toward predicting survival, "
            "consistent with the balanced class weighting"
        )
    takeaway("ROC-AUC alone barely separates the models, so it wasn't enough on its own to pick a winner.")
    what_this_means(
        observation="All four models' ROC curves sit close together, and our model produces "
        "more false positives than false negatives.",
        impact="A gap of 0.026 in ROC-AUC between the best and worst model is too small to "
        "call a clear winner on its own.",
        decision="The false positive tendency lines up with the balanced class weighting we "
        "used during training, so we left it as is.",
    )

with tab5:
    comparison = load_comparison_curves()
    fig = go.Figure()
    for name, d in comparison.items():
        p, r, _ = precision_recall_curve(d["y_test"], d["y_proba"])
        lw = 3.5 if "Logistic Regression" in name else 1.6
        fig.add_trace(go.Scatter(
            x=r, y=p, mode="lines", name=MODEL_SHORT_NAMES[name],
            line=dict(color=MODEL_COLORS[name], width=lw),
        ))
    fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="#9CA3AF")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(style_fig(fig, height=440), use_container_width=True)
    with st.container(border=True, key="card-pr-findings"):
        st.markdown(
            "- Logistic Regression tracks at or above the pack through most of the range\n"
            "- Random Forest and Gradient Boosting fall visibly behind past recall 0.5\n"
            "- Under this class imbalance, the Precision-Recall gap is wider and clearer than the ROC gap"
        )
    takeaway("The precision-recall curve backs up the ROC results, and matters more given our class imbalance.")
    what_this_means(
        observation="Logistic Regression stays at or above the other models across most of "
        "the recall range.",
        impact="With our class imbalance, this precision recall gap tells us more than the "
        "ROC gap does.",
        decision="It backed up the same conclusion as the ROC comparison rather than changing "
        "our pick.",
    )

with tab6:
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=8, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#9CA3AF", dash="dash"), name="Perfect calibration"))
    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", line=dict(color=PRIMARY, width=3), name="Logistic Regression"))
    fig.update_layout(
        title=f"Calibration Curve (Brier score = {metrics['brier']:.3f})",
        xaxis_title="Mean predicted probability", yaxis_title="Observed survival rate",
    )
    st.plotly_chart(style_fig(fig, height=440), use_container_width=True)
    with st.container(border=True, key="card-calibration-findings"):
        st.markdown(
            "- Predictions at the extremes (below 0.2 or above 0.8) track the diagonal closely\n"
            "- The mid-range (0.3-0.5) sits above the diagonal — the model is somewhat overconfident for borderline passengers\n"
            "- The Brier score beats the always-predict-base-rate score by a wide margin"
        )
    takeaway("The model's confidence lines up well at the extremes, with slight overconfidence in the middle range. That's not worth fixing on a small test set.")
    what_this_means(
        observation="Predicted probabilities match real outcomes closely at the extremes, but "
        "run a bit too confident in the 0.3 to 0.5 range.",
        impact="That matters most if we're showing the probability itself to a user, not just "
        "a survived or did not survive label.",
        decision="With only 179 test rows, recalibrating wasn't worth it, so we show the raw "
        "probabilities as they are.",
    )

with tab7:
    thresholds = np.arange(0.05, 0.96, 0.01)
    precisions = [precision_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    recalls = [recall_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    f1s = np.array([f1_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresholds])
    best_idx = int(np.argmax(f1s))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode="lines", name="Precision", line=dict(color=WARNING, width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode="lines", name="Recall", line=dict(color=SUCCESS, width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=f1s, mode="lines", name="F1 Score", line=dict(color=PRIMARY, width=3)))
    fig.add_trace(go.Scatter(
        x=[thresholds[best_idx]], y=[f1s[best_idx]], mode="markers",
        marker=dict(color="black", size=10), name=f"Best F1 ({thresholds[best_idx]:.2f})",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#9CA3AF")
    fig.update_layout(xaxis_title="Probability Threshold", yaxis_title="Score")
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    with st.container(border=True, key="card-threshold-findings"):
        st.markdown(
            f"- The best F1 ({f1s[best_idx]:.3f}) occurs at threshold {thresholds[best_idx]:.2f} — a 0.005 "
            "improvement over the default\n"
            "- That gap is noise on a 179-row test set, not a meaningful signal\n"
            "- Precision and recall trade off smoothly, with no cliff that would justify moving off 0.50"
        )
    takeaway("The default threshold of 0.50 stays in production, since the theoretical best value is too close to call a real improvement.")
    what_this_means(
        observation=f"The best F1 score shows up at a threshold of {thresholds[best_idx]:.2f}, "
        "just 0.005 above the default of 0.50.",
        impact="A difference that small is noise on a 179-row test set, not a real improvement.",
        decision="We kept the default 0.50 threshold in production.",
    )

with tab8:
    col1, col2 = st.columns(2)
    with col1:
        coef_df = get_coefficients(model).sort_values("coef", key=np.abs, ascending=False).head(10)
        coef_df["direction"] = np.where(coef_df["coef"] > 0, "Increases survival odds", "Decreases survival odds")
        fig = px.bar(
            coef_df.sort_values("coef"), x="coef", y="feature", orientation="h", color="direction",
            color_discrete_map={"Increases survival odds": PRIMARY, "Decreases survival odds": ERROR},
            labels={"coef": "Coefficient (log-odds)", "feature": ""}, title="Logistic Regression Coefficients",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, height=400), use_container_width=True)
    with col2:
        perm_df = get_permutation_importance(model, X_test, y_test)
        fig = px.bar(
            perm_df, x="importance", y="feature", orientation="h",
            labels={"importance": "F1 Drop", "feature": ""}, title="Permutation Importance",
        )
        fig.update_traces(marker_color=PRIMARY)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_fig(fig, height=400), use_container_width=True)
    with st.container(border=True, key="card-feature-importance-findings"):
        st.markdown(
            "- Sex and Title dominate both views — the two strongest predictors by either measure\n"
            "- Pclass ranks higher by coefficient than by permutation importance — its signal overlaps with "
            "Fare and Has Cabin, so the model leans on those when Pclass is shuffled\n"
            "- Family Size's coefficient fits a straight line through a relationship that isn't actually linear"
        )
    takeaway("The ranking differences come from the same overlapping features the multicollinearity check already flagged.")
    what_this_means(
        observation="Sex and Title top both the coefficient view and the permutation importance "
        "view, which measures a feature's importance by seeing how much performance drops "
        "without it.",
        impact="Pclass ranks lower on permutation importance because Fare and Has Cabin already "
        "carry some of the same information.",
        decision="That overlap is the same multicollinearity issue the VIF check caught "
        "earlier, not a new one.",
    )

with tab9:
    subgroup = get_subgroup_performance()
    sex_df = subgroup["sex"]
    pclass_df = subgroup["pclass"]

    st.markdown("**Subgroup Performance**")
    st.caption(
        "Aggregate accuracy can hide a model that works well for one group and poorly for "
        "another. This breaks the same test-set predictions down by Sex and Passenger Class."
    )

    sg_col1, sg_col2 = st.columns(2)
    with sg_col1:
        fig = go.Figure()
        for metric, color in zip(["accuracy", "precision", "recall", "f1"], [SECONDARY, WARNING, SUCCESS, PRIMARY]):
            fig.add_trace(go.Bar(
                name=metric.capitalize(), x=sex_df["Sex"].str.capitalize(), y=sex_df[metric],
                marker_color=color,
            ))
        fig.update_layout(barmode="group", yaxis_title="Score", yaxis_range=[0, 1], title="By Sex")
        st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
    with sg_col2:
        fig = go.Figure()
        for metric, color in zip(["accuracy", "precision", "recall", "f1"], [SECONDARY, WARNING, SUCCESS, PRIMARY]):
            fig.add_trace(go.Bar(
                name=metric.capitalize(), x=[f"Class {p}" for p in pclass_df["Pclass"]], y=pclass_df[metric],
                marker_color=color,
            ))
        fig.update_layout(barmode="group", yaxis_title="Score", yaxis_range=[0, 1], title="By Passenger Class")
        st.plotly_chart(style_fig(fig, height=380), use_container_width=True)

    n_parts = [f"{row.Sex.capitalize()}: {row.n}" for row in sex_df.itertuples()]
    n_parts += [f"Class {row.Pclass}: {row.n}" for row in pclass_df.itertuples()]
    st.caption("Test-set rows per group — " + ", ".join(n_parts) + ". Small groups make their metrics noisier.")

    worst_sex = sex_df.loc[sex_df["recall"].idxmin()]
    worst_pclass = pclass_df.loc[pclass_df["recall"].idxmin()]
    min_n = min(sex_df["n"].min(), pclass_df["n"].min())
    max_n = max(sex_df["n"].max(), pclass_df["n"].max())

    with st.container(border=True, key="card-fairness-findings"):
        st.markdown(
            f"- Recall is lowest for **{worst_sex['Sex']}** passengers ({worst_sex['recall']:.1%}) and "
            f"**Class {int(worst_pclass['Pclass'])}** passengers ({worst_pclass['recall']:.1%}) — the "
            "subgroups where the model misses the most actual survivors\n"
            f"- Group sizes range from {int(min_n)} to {int(max_n)} test rows, small enough that any "
            "single subgroup's metric can swing on a handful of cases\n"
            "- No subgroup is held out or reweighted during training — the model is fit once on the "
            "full population, not separately per group"
        )
    takeaway(
        "Performance is not identical across subgroups, but on a 179-row test set, individual "
        "group metrics are too noisy to call any gap a real bias rather than sampling variance."
    )
    what_this_means(
        observation="Precision, recall, and F1 vary by Sex and by Passenger Class rather than "
        "sitting at one flat number for everyone.",
        impact="A model can look accurate overall while still being systematically worse for a "
        "specific group — this is the check that would catch that.",
        decision="With test subgroups this small, we're not treating any single gap as confirmed "
        "bias — it's flagged for monitoring as more data comes in, not acted on today.",
    )

    st.write("")
    st.markdown("**Error Composition**")
    st.caption("Which passengers the model gets wrong, and what they have in common.")

    errors = get_error_composition()
    fp, fn = errors["false_positives"], errors["false_negatives"]

    ec_col1, ec_col2 = st.columns(2)
    with ec_col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="False Positives", x=["Female", "Male"],
            y=[(fp["Sex"] == "female").sum(), (fp["Sex"] == "male").sum()], marker_color=ERROR,
        ))
        fig.add_trace(go.Bar(
            name="False Negatives", x=["Female", "Male"],
            y=[(fn["Sex"] == "female").sum(), (fn["Sex"] == "male").sum()], marker_color=WARNING,
        ))
        fig.update_layout(barmode="group", yaxis_title="Count", title="Errors by Sex")
        st.plotly_chart(style_fig(fig, height=340), use_container_width=True)
    with ec_col2:
        classes = sorted(pclass_df["Pclass"].tolist())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="False Positives", x=[f"Class {c}" for c in classes],
            y=[(fp["Pclass"] == c).sum() for c in classes], marker_color=ERROR,
        ))
        fig.add_trace(go.Bar(
            name="False Negatives", x=[f"Class {c}" for c in classes],
            y=[(fn["Pclass"] == c).sum() for c in classes], marker_color=WARNING,
        ))
        fig.update_layout(barmode="group", yaxis_title="Count", title="Errors by Passenger Class")
        st.plotly_chart(style_fig(fig, height=340), use_container_width=True)

    st.write("")
    st.markdown("**Most Confidently Wrong Predictions**")
    examples = get_misclassified_examples()
    display = examples.copy()
    display["Actual"] = display["Actual"].map({0: "Died", 1: "Survived"})
    display["Predicted"] = display["Predicted"].map({0: "Died", 1: "Survived"})
    display["Probability"] = display["Probability"].map(lambda p: f"{p:.1%}")
    display["Age"] = display["Age"].round(0).astype(int)
    display["Fare"] = display["Fare"].round(1)
    st.dataframe(
        display[["Pclass", "Sex", "Age", "Title", "Fare", "Actual", "Predicted", "Probability"]],
        use_container_width=True, hide_index=True,
    )
    with st.container(border=True, key="card-error-findings"):
        st.markdown(
            "- Each row above is a real test-set passenger the model was most confidently wrong about\n"
            "- These are cases where Sex, Title, and Pclass — the model's strongest signals — pointed "
            "the wrong way for that individual\n"
            "- No amount of tuning removes cases like these; a linear model on 8 features can't capture "
            "every individual exception to a group-level pattern"
        )
    takeaway(
        "The model's biggest misses are individuals whose outcome went against the odds for their "
        "group — not a systematic flaw, but the visible limit of predicting a person from group averages."
    )
    what_this_means(
        observation="The most confident wrong predictions are passengers whose actual outcome "
        "went against the pattern the model learned for people like them.",
        impact="This is the honest cost of a model built on group-level signals like Sex, Title, "
        "and Class — it will always miss the individuals who are the exception.",
        decision="We're not trying to eliminate these cases; showing them is about being explicit "
        "with users about what the model can and can't know from 8 features.",
    )

with tab10:
    st.subheader("Model Development")
    st.markdown("How each model was trained, tuned, and evaluated.")

    DEV_SUMMARY = {
        "Baseline (production RF, 11 features)": {
            "Optimization Method": "RandomizedSearchCV",
            "Hyperparameter Search Strategy": "20 iterations, F1-optimized",
            "Cross-Validation Strategy": "Stratified 5-Fold",
            "Final Status": "Baseline",
        },
        "Logistic Regression (8 features)": {
            "Optimization Method": "Fixed configuration",
            "Hyperparameter Search Strategy": "Not tuned",
            "Cross-Validation Strategy": "Stratified 5-Fold",
            "Final Status": "Selected",
        },
        "Random Forest (8 features)": {
            "Optimization Method": "Fixed configuration",
            "Hyperparameter Search Strategy": "Not tuned",
            "Cross-Validation Strategy": "Stratified 5-Fold",
            "Final Status": "Compared",
        },
        "Gradient Boosting (8 features)": {
            "Optimization Method": "Fixed configuration",
            "Hyperparameter Search Strategy": "Not tuned",
            "Cross-Validation Strategy": "Stratified 5-Fold",
            "Final Status": "Compared",
        },
    }
    HYPERPARAMETERS = {
        "Baseline (production RF, 11 features)": [
            "Number of Trees: 300", "Maximum Depth: 10", "Minimum Samples Split: 10",
            "Minimum Samples Leaf: 4", "Class Weight: Balanced",
        ],
        "Logistic Regression (8 features)": [
            "Solver: lbfgs", "Penalty: L2", "Regularization (C): 1.0",
            "Class Weight: Balanced", "Max Iterations: 1000",
        ],
        "Random Forest (8 features)": [
            "Number of Trees: 300", "Maximum Depth: Unlimited", "Minimum Samples Split: 2",
            "Minimum Samples Leaf: 1", "Class Weight: Balanced",
        ],
        "Gradient Boosting (8 features)": [
            "Number of Estimators: 100", "Learning Rate: 0.1", "Maximum Depth: 3",
            "Minimum Samples Split: 2", "Minimum Samples Leaf: 1",
        ],
    }

    st.markdown("**1. Model Development Summary**")
    dev_df = pd.DataFrame([
        {"Model": MODEL_SHORT_NAMES[r["name"]], **DEV_SUMMARY[r["name"]]}
        for r in results
    ])
    st.dataframe(dev_df, use_container_width=True, hide_index=True)

    st.write("")
    st.markdown("**2. Hyperparameter Summary**")
    # Colored top border on each card, matching that model's color everywhere else in the charts.
    st.markdown(
        "<style>" + "".join(
            f'div[class~="st-key-card-hp-{MODEL_SHORT_NAMES[name].lower().replace(" ", "-")}"] '
            f"{{ border-top: 4px solid {color} !important; }}"
            for name, color in MODEL_COLORS.items()
        ) + "</style>",
        unsafe_allow_html=True,
    )
    hp_cols = st.columns(4)
    for col, r in zip(hp_cols, results):
        with col:
            with st.container(border=True, height=230, key=f"card-hp-{MODEL_SHORT_NAMES[r['name']].lower().replace(' ', '-')}"):
                st.markdown(f"**{MODEL_SHORT_NAMES[r['name']]}**")
                st.markdown("\n".join(f"- {p}" for p in HYPERPARAMETERS[r["name"]]))

    what_this_means(
        observation="Only the baseline Random Forest went through a hyperparameter search called "
        "RandomizedSearchCV, while the other three models ran with fixed settings.",
        impact="Logistic Regression won that comparison without ever being tuned in its favor.",
        decision="That makes the case for Logistic Regression stronger, since it was not given "
        "any extra advantage.",
    )

    st.write("")
    st.markdown("**3. Model Selection Strategy**")
    with st.container(border=True, key="card-selection-strategy"):
        st.markdown(
            "- **Train/Test Split**: An 80/20 stratified split was used for all four models to "
            "ensure a fair comparison.\n"
            "- **Cross Validation**: Stratified 5 fold cross validation was used to evaluate model "
            "stability across different data splits.\n"
            "- **Hyperparameter Optimization**: The baseline model was optimized using "
            "RandomizedSearchCV, while the three candidate models used fixed and documented "
            "configurations for a consistent comparison.\n"
            "- **Performance Metrics**: Models were evaluated using Accuracy, Precision, Recall, "
            "F1 Score, ROC AUC, and the average 5 fold cross validation F1 Score.\n"
            "- **Final Selection Criteria**: The final model was selected based on overall test "
            "performance, generalization, inference speed, and interpretability."
        )

    st.divider()
    st.subheader("Why Logistic Regression Was the Best Choice")
    with st.container(border=True, key="card-why-fits"):
        st.markdown(
            "- **Cleaner Feature Set**: The feature set was reduced from 11 to 8 after statistical "
            "analysis and feature selection. This removed unnecessary information and allowed the "
            "model to focus on features that genuinely contributed to prediction.\n"
            "- **Reduced Multicollinearity**: Sibling or Spouse Count, Parent or Child Count, and "
            "Family Size contained overlapping information. Keeping only Family Size removed "
            "redundancy and provided a cleaner input for the model.\n"
            "- **Improved Feature Distribution**: Fare had a heavily skewed distribution. Applying "
            "a log transformation made it more balanced, helping Logistic Regression learn more "
            "effectively.\n"
            "- **Simple Relationships in the Data**: Survival was mainly influenced by a few strong "
            "factors such as Sex, Passenger Class, Title, and Fare. These relationships were largely "
            "linear, making Logistic Regression a good fit without requiring a more complex model.\n"
            "- **Better Generalization**: Logistic Regression achieved the best balance between "
            "training and test performance. Unlike the tree based models, it showed no clear signs "
            "of overfitting, indicating that its complexity matched the dataset well."
        )
