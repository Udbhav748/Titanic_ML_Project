"""Cached data/model/artifact loaders shared across every dashboard page."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from analysis.stage4_model_comparison import fit_all_for_comparison, load_candidate_split

PRODUCTION_FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked", "HasCabin", "FamilySize", "Title"]


@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "data" / "train.csv")


@st.cache_resource
def load_preprocessor():
    return joblib.load(BASE_DIR / "artifacts" / "preprocessing.pkl")


@st.cache_data
def load_transformed() -> pd.DataFrame:
    preprocessor = load_preprocessor()
    return preprocessor.transform(load_raw())


@st.cache_resource
def load_model_v2():
    return joblib.load(BASE_DIR / "model" / "model_v2.pkl")


@st.cache_resource
def load_model_v1():
    return joblib.load(BASE_DIR / "model" / "model_v1.pkl")


@st.cache_data
def load_feature_schema() -> dict:
    with open(BASE_DIR / "artifacts" / "feature_schema.json") as f:
        return json.load(f)


@st.cache_data
def load_stage4_results() -> list:
    with open(BASE_DIR / "reports" / "stage4_model_comparison.json") as f:
        return json.load(f)


def get_coefficients(model) -> pd.DataFrame:
    """Feature name -> logistic regression coefficient, in the pipeline's transformed feature space."""
    names = [f.split("__", 1)[-1] for f in model.named_steps["preprocessor"].get_feature_names_out()]
    coefs = model.named_steps["classifier"].coef_[0]
    return pd.DataFrame({"feature": names, "coef": coefs})


@st.cache_resource
def load_comparison_curves() -> dict:
    """Cached wrapper around fit_all_for_comparison — fits the baseline and all
    three candidates once per server process (the baseline's RandomizedSearchCV
    is too expensive to redo on every page view) and reuses the result for every
    ROC/Precision-Recall overlay on the Model Performance page."""
    return fit_all_for_comparison()


@st.cache_data
def get_permutation_importance(_model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Permutation importance for the production model, cached by test data identity."""
    perm = permutation_importance(_model, X_test, y_test, n_repeats=30, random_state=42, scoring="f1")
    return pd.DataFrame(
        {"feature": X_test.columns, "importance": perm.importances_mean}
    ).sort_values("importance", ascending=False)


@st.cache_data
def load_train_reference() -> tuple:
    """The 712-row training split (X_train, y_train) — the historical population
    used for similarity search and population statistics on the Live Prediction page."""
    X_train, _, y_train, _ = load_candidate_split()
    return X_train, y_train


@st.cache_resource
def get_transformed_train_matrix() -> np.ndarray:
    """Training features transformed through the production model's own fitted
    preprocessor — the exact numeric space used for similarity search, never refit."""
    X_train, _ = load_train_reference()
    model = load_model_v2()
    matrix = model.named_steps["preprocessor"].transform(X_train)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def _vectorized_auc(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """ROC-AUC per row of two (n_bootstrap, n) matrices, via the rank-sum
    (Mann-Whitney U) identity — equivalent to sklearn.roc_auc_score but
    computed for every resample in one vectorized pass instead of one
    Python-level call per resample."""
    order = np.argsort(yp, axis=1)
    ranks = np.empty_like(order, dtype=float)
    row_idx = np.arange(yp.shape[0])[:, None]
    ranks[row_idx, order] = np.arange(1, yp.shape[1] + 1)
    n_pos = yt.sum(axis=1)
    n_neg = yt.shape[1] - n_pos
    sum_ranks_pos = (ranks * yt).sum(axis=1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@st.cache_data
def get_bootstrap_ci(n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Percentile bootstrap on each model's held-out test predictions —
    quantifies exactly how much sampling noise a 179-row test set produces,
    instead of just asserting the set is small. The same resample indices
    are reused across every model (all four share identical test rows —
    see analysis/stage4_model_comparison.py), so this is a paired bootstrap,
    not four independent ones. Fully vectorized: every resample for every
    metric is computed in one array pass rather than one sklearn call per
    resample, which is what makes 1,000 resamples x 4 models tractable
    inside a Streamlit page load."""
    comparison = load_comparison_curves()
    names = list(comparison.keys())
    n = len(comparison[names[0]]["y_test"])
    rng = np.random.default_rng(42)
    idx_matrix = rng.integers(0, n, size=(n_bootstrap, n))
    alpha = (1 - ci) / 2 * 100

    result = {}
    for name in names:
        y_test = np.asarray(comparison[name]["y_test"])
        y_proba = np.asarray(comparison[name]["y_proba"])
        yt = y_test[idx_matrix]
        yp = y_proba[idx_matrix]
        pred = (yp >= 0.5).astype(int)

        tp = ((pred == 1) & (yt == 1)).sum(axis=1)
        fp = ((pred == 1) & (yt == 0)).sum(axis=1)
        fn = ((pred == 0) & (yt == 1)).sum(axis=1)
        tn = ((pred == 0) & (yt == 0)).sum(axis=1)

        valid = (yt.min(axis=1) != yt.max(axis=1))  # degenerate resample — AUC undefined
        accuracy = (tp + tn) / yt.shape[1]
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        f1 = np.divide(
            2 * precision * recall, precision + recall,
            out=np.zeros_like(precision), where=(precision + recall) > 0,
        )
        auc = _vectorized_auc(yt[valid], yp[valid])

        metrics = {
            "accuracy": accuracy[valid], "precision": precision[valid],
            "recall": recall[valid], "f1": f1[valid], "roc_auc": auc,
        }
        result[name] = {
            k: (float(np.percentile(v, alpha)), float(np.percentile(v, 100 - alpha)))
            for k, v in metrics.items()
        }
    return result


@st.cache_resource
def get_shap_explainer():
    """LinearExplainer for the production model's classifier, using the full
    transformed training set as background — SHAP values are exact for a
    linear model given this background distribution (they sum exactly to
    the model's own decision_function, not an approximation)."""
    import shap

    model = load_model_v2()
    background = get_transformed_train_matrix()
    masker = shap.maskers.Independent(background, max_samples=len(background))
    return shap.LinearExplainer(model.named_steps["classifier"], masker)


@st.cache_data
def get_logit_strength_cutoffs() -> tuple:
    """Tertile cutoffs of |log-odds| on the held-out test set — buckets a
    prediction's decision strength against the model's own real score
    distribution instead of an arbitrary fixed threshold."""
    _, X_test, _, _ = load_candidate_split()
    model = load_model_v2()
    abs_logits = np.abs(model.decision_function(X_test))
    low, high = np.percentile(abs_logits, [33.33, 66.67])
    return float(low), float(high)


@st.cache_data
def get_population_reference() -> dict:
    """Real dataset statistics used to phrase feature contributions and build
    realistic counterfactual candidate values — never invented numbers."""
    raw = load_raw()
    fare = raw["Fare"].dropna()
    age = raw["Age"].dropna()
    family_size = raw["SibSp"] + raw["Parch"] + 1
    return {
        "age_median": float(age.median()),
        "age_q25": float(age.quantile(0.25)),
        "age_q75": float(age.quantile(0.75)),
        "fare_median": float(fare.median()),
        "fare_q25": float(fare.quantile(0.25)),
        "fare_q75": float(fare.quantile(0.75)),
        "family_median": float(family_size.median()),
        "family_max": int(family_size.max()),
    }


@st.cache_data
def get_subgroup_performance() -> dict:
    """Model accuracy/precision/recall/F1 broken out by Sex and by Pclass —
    whether the model is equally reliable across subgroups, not just accurate
    on average. Aggregate metrics can hide a model that works well for one
    group and poorly for another."""
    X_test, y_test, y_pred, _, _ = evaluate_v2_on_test()
    frame = X_test.copy()
    frame["y_true"] = y_test.values
    frame["y_pred"] = y_pred

    def _by(col: str) -> pd.DataFrame:
        rows = []
        for value, g in frame.groupby(col):
            rows.append({
                col: value,
                "n": len(g),
                "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
                "precision": precision_score(g["y_true"], g["y_pred"], zero_division=0),
                "recall": recall_score(g["y_true"], g["y_pred"], zero_division=0),
                "f1": f1_score(g["y_true"], g["y_pred"], zero_division=0),
            })
        return pd.DataFrame(rows)

    return {"sex": _by("Sex"), "pclass": _by("Pclass")}


@st.cache_data
def get_error_composition() -> dict:
    """False positives and false negatives on the test set, with the full
    feature row attached — which passengers the model got wrong, and what
    subgroups (Sex, Pclass, Title) its mistakes cluster in."""
    X_test, y_test, y_pred, y_proba, _ = evaluate_v2_on_test()
    frame = X_test.copy()
    frame["Actual"] = y_test.values
    frame["Predicted"] = y_pred
    frame["Probability"] = y_proba
    fp = frame[(frame["Actual"] == 0) & (frame["Predicted"] == 1)]
    fn = frame[(frame["Actual"] == 1) & (frame["Predicted"] == 0)]
    return {"false_positives": fp, "false_negatives": fn}


@st.cache_data
def get_misclassified_examples(top_n: int = 8) -> pd.DataFrame:
    """The test set's most confidently wrong predictions, ranked by how far
    the model's probability missed the true outcome — the cases worth
    inspecting individually, not just counting."""
    X_test, y_test, y_pred, y_proba, _ = evaluate_v2_on_test()
    frame = X_test.copy()
    frame["Actual"] = y_test.values
    frame["Predicted"] = y_pred
    frame["Probability"] = y_proba
    wrong = frame[frame["Actual"] != frame["Predicted"]].copy()
    wrong["miss_margin"] = np.where(
        wrong["Actual"] == 1, 1 - wrong["Probability"], wrong["Probability"]
    )
    return wrong.sort_values("miss_margin", ascending=False).head(top_n).drop(columns="miss_margin")


@st.cache_data
def evaluate_v2_on_test():
    """Test-set predictions for model_v2, reused by Model Performance/Live Prediction."""
    X_train, X_test, y_train, y_test = load_candidate_split()
    model = load_model_v2()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "brier": brier_score_loss(y_test, y_proba),
    }
    return X_test, y_test, y_pred, y_proba, metrics
