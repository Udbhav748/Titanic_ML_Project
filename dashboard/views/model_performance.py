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
    get_coefficients,
    get_permutation_importance,
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Model Comparison", "Cross-Validation", "Train vs Test", "ROC & Confusion Matrix",
    "Precision-Recall", "Calibration", "Threshold Analysis", "Feature Importance",
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
    fig = go.Figure()
    for r in results:
        fig.add_trace(go.Bar(
            name=MODEL_SHORT_NAMES[r["name"]], x=metric_labels, y=[r[k] for k in metric_keys],
            marker_color=MODEL_COLORS[r["name"]],
        ))
    fig.update_layout(barmode="group", yaxis_title="Score", yaxis_range=[0, 1])
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    with st.container(border=True, key="card-model-comparison-findings"):
        st.markdown(
            "- Logistic Regression leads on 4 of 5 metrics — accuracy, recall, F1, and ROC-AUC\n"
            "- Gradient Boosting edges it only on precision (0.783 vs. 0.760)\n"
            "- Random Forest is the weakest model on every metric, despite sharing the same 8-feature schema"
        )
    takeaway("Performance alone favors Logistic Regression, but the margin is modest — stability and generalization matter just as much.")

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
    takeaway("Tight variance without a competitive score isn't real stability — it's consistency around a weaker answer.")

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
    takeaway("Only Logistic Regression generalizes with no overfitting signature — the rest memorize to varying degrees.")

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
    takeaway("ROC-AUC alone would have made this a close call — a 0.026 spread is not a decisive signal by itself.")

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
    takeaway("PR analysis reinforces the ROC comparison and carries more weight under a 62/38 class imbalance.")

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
    takeaway("Well-calibrated at the extremes, mildly overconfident mid-range — not worth recalibrating on a 179-row test set.")

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
    takeaway("The default 0.50 threshold stays in production — the theoretical optimum is within measurement noise.")

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
    takeaway("Ranking differences trace back to the same correlated features the multicollinearity analysis already flagged.")

st.divider()
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

st.write("")
st.markdown("**3. Model Selection Strategy**")
with st.container(border=True, key="card-selection-strategy"):
    st.markdown(
        "- **Train/Test Split**: 80/20 stratified split, identical across all four models\n"
        "- **Cross-Validation Method**: Stratified 5-fold cross-validation\n"
        "- **Hyperparameter Optimization**: RandomizedSearchCV for the baseline; fixed, documented "
        "configurations for the three candidates\n"
        "- **Performance Metrics**: Accuracy, Precision, Recall, F1, and ROC-AUC on the test set, plus "
        "5-fold CV F1 for stability\n"
        "- **Final Selection Criteria**: Best test performance, no overfitting signature, faster inference, "
        "and full interpretability"
    )

st.divider()
st.subheader("Why Logistic Regression Fits This Data")
with st.container(border=True, key="card-why-fits"):
    st.markdown(
        "- **Feature set reduced from 11 to 8** — every remaining feature carries real signal, and a linear "
        "model has no way to route around noisy ones the way a tree can\n"
        "- **Multicollinearity removed** — sibling/spouse count, parent/child count, and family size were "
        "perfectly collinear; keeping only Family Size gives the model one clean coefficient instead of three "
        "competing for the same variance\n"
        "- **Fare's skew corrected** — a log transform fixed Fare's heavy right skew, satisfying the roughly-linear "
        "relationship Logistic Regression assumes between a feature and log-odds\n"
        "- **A simple decision boundary was enough** — survival is driven by a handful of dominant factors "
        "(Sex, Passenger Class, Title, Fare), not subtle interactions, so a linear boundary fit as well as a "
        "non-linear one\n"
        "- **Confirmed by generalization** — it is the only model with no overfitting signature, evidence that "
        "its capacity actually matched the complexity in the data"
    )
