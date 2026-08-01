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
