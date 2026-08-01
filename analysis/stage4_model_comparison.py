"""
Stage 4 — Model Retraining & Comparison.

A controlled experiment: does the Stage 3 production schema (8 features,
leakage-safe preprocessing) beat the currently deployed model (11 features,
model/train.py's existing pipeline)?

Every model shares the same row-level train/test split (train_test_split on
the same 891-row frame, same random_state, same stratify column — sklearn's
split only depends on row count/order and the stratify labels, not on which
columns X has, so baseline and candidates get identical test rows even
though their feature matrices differ). Every candidate shares the same
StratifiedKFold, the same random_state, and no per-algorithm hyperparameter
search — only the learning algorithm changes among them. The baseline is
tuned via model/train.py's own RandomizedSearchCV, because reproducing
"the deployed model" means reproducing how it was actually built.

Run: python analysis/stage4_model_comparison.py
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model.train import build_preprocessor as build_baseline_preprocessor  # noqa: E402
from model.train import load_and_prepare_data as load_baseline_data  # noqa: E402
from model.train import tune_random_forest as tune_baseline_rf  # noqa: E402

DATA_PATH = BASE_DIR / "data" / "train.csv"
PREPROCESSING_PKL_PATH = BASE_DIR / "artifacts" / "preprocessing.pkl"
REPORT_PATH = BASE_DIR / "reports" / "stage4_model_comparison.md"

NEW_SCHEMA_FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked", "HasCabin", "FamilySize", "Title"]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_candidate_preprocessor() -> ColumnTransformer:
    """8-feature Stage 3 schema. No imputers: TitanicPreprocessor already
    filled Age/Embarked upstream. Fare gets log1p ahead of scaling —
    monotonic, harmless to trees, fixes the skew for Logistic Regression."""
    fare_pipeline = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(transformers=[
        ("fare", fare_pipeline, ["Fare"]),
        ("num", StandardScaler(), ["Age", "FamilySize"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Sex", "Embarked", "Pclass", "Title"]),
        ("pass", "passthrough", ["HasCabin"]),
    ])


def load_baseline_split() -> tuple:
    X, y = load_baseline_data()
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def load_candidate_split() -> tuple:
    preprocessor = joblib.load(PREPROCESSING_PKL_PATH)
    raw = pd.read_csv(DATA_PATH)
    train_raw, test_raw = train_test_split(
        raw, test_size=0.2, random_state=RANDOM_STATE, stratify=raw["Survived"]
    )
    train_full = preprocessor.transform(train_raw)
    test_full = preprocessor.transform(test_raw)
    X_train = train_full[NEW_SCHEMA_FEATURES]
    X_test = test_full[NEW_SCHEMA_FEATURES]
    y_train, y_test = train_raw["Survived"], test_raw["Survived"]
    return X_train, X_test, y_train, y_test


def evaluate(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test, n_features: int) -> dict:
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=CV, scoring="f1")

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    train_pred = pipeline.predict(X_train)
    train_f1 = f1_score(y_train, train_pred)

    t0 = time.perf_counter()
    test_pred = pipeline.predict(X_test)
    predict_time_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "name": name,
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred),
        "recall": recall_score(y_test, test_pred),
        "f1": f1_score(y_test, test_pred),
        "roc_auc": roc_auc_score(y_test, test_proba),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores.tolist(),
        "train_f1": train_f1,
        "train_time_s": train_time,
        "predict_latency_ms": predict_time_ms,
        "n_features": n_features,
    }


def run_baseline() -> dict:
    X_train, X_test, y_train, y_test = load_baseline_split()
    preprocessor = build_baseline_preprocessor()

    best_rf = tune_baseline_rf(X_train, y_train, preprocessor)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", best_rf)])

    result = evaluate("Baseline (production RF, 11 features)", pipeline, X_train, y_train, X_test, y_test,
                       n_features=X_train.shape[1])
    result["schema"] = "current (11 features, full-dataset-fit imputation)"
    return result


def run_candidates() -> list:
    X_train, X_test, y_train, y_test = load_candidate_split()
    preprocessor = build_candidate_preprocessor()

    models = {
        "Logistic Regression (8 features)": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest (8 features)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Gradient Boosting (8 features)": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = []
    for name, clf in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        result = evaluate(name, pipeline, X_train, y_train, X_test, y_test, n_features=X_train.shape[1])
        result["schema"] = "proposed (8 features, leakage-safe imputation, Fare log1p)"
        results.append(result)
    return results


def fit_all_for_comparison() -> dict:
    """Fit the baseline and all three candidates once, returning each model's
    held-out probabilities on its own (row-identical) test split. Not used by
    main() — this exists for the Stage 4/5 notebooks' cross-model curve
    overlays (ROC, Precision-Recall), which need probabilities main() doesn't
    persist to the JSON report."""
    comparison = {}

    X_train_b, X_test_b, y_train_b, y_test_b = load_baseline_split()
    baseline_pre = build_baseline_preprocessor()
    best_rf = tune_baseline_rf(X_train_b, y_train_b, baseline_pre)
    baseline_pipe = Pipeline([("preprocessor", baseline_pre), ("classifier", best_rf)])
    baseline_pipe.fit(X_train_b, y_train_b)
    comparison["Baseline (production RF, 11 features)"] = {
        "y_test": y_test_b, "y_proba": baseline_pipe.predict_proba(X_test_b)[:, 1],
    }

    X_train_c, X_test_c, y_train_c, y_test_c = load_candidate_split()
    candidate_pre = build_candidate_preprocessor()
    candidate_models = {
        "Logistic Regression (8 features)": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest (8 features)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Gradient Boosting (8 features)": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    for name, clf in candidate_models.items():
        pipe = Pipeline([("preprocessor", candidate_pre), ("classifier", clf)])
        pipe.fit(X_train_c, y_train_c)
        comparison[name] = {"y_test": y_test_c, "y_proba": pipe.predict_proba(X_test_c)[:, 1]}

    return comparison


def render_metrics_table(results: list) -> str:
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Mean (F1) | CV Std | Train Time (s) | Predict Latency (ms/row) | Features |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {r['roc_auc']:.3f} | {r['cv_mean']:.3f} | {r['cv_std']:.3f} | "
            f"{r['train_time_s']:.3f} | {r['predict_latency_ms']:.4f} | {r['n_features']} |"
        )
    return "\n".join(lines)


def render_overfitting_table(results: list) -> str:
    lines = [
        "| Model | Train F1 | CV Mean F1 | Test F1 | Train − Test Gap | Read |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        gap = r["train_f1"] - r["f1"]
        if gap > 0.12:
            read = "Overfitting"
        elif r["train_f1"] < 0.55:
            read = "Underfitting"
        else:
            read = "Stable"
        lines.append(
            f"| {r['name']} | {r['train_f1']:.3f} | {r['cv_mean']:.3f} | {r['f1']:.3f} | "
            f"{gap:+.3f} | {read} |"
        )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Tuning and evaluating baseline (this runs RandomizedSearchCV, ~10-20s)...")
    baseline = run_baseline()

    print("Evaluating candidates on the proposed schema...")
    candidates = run_candidates()

    all_results = [baseline] + candidates

    print("\n" + render_metrics_table(all_results))
    print("\n" + render_overfitting_table(all_results))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(REPORT_PATH.with_suffix(".json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[Raw results written to {REPORT_PATH.with_suffix('.json')}]")


if __name__ == "__main__":
    main()
