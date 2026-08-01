"""
Promote the Stage 4/5-validated Logistic Regression model to production.

Trains the exact pipeline evaluated in analysis/stage4_model_comparison.py
(same split, same schema, same config) and saves it as model/model_v2.pkl,
so its metrics match what Stage 4/5 already reported byte-for-byte. Also
bumps artifacts/feature_schema.json and preprocessing_metadata.json to
reflect the new production schema — the "promotion" step from Stage 6's
migration plan, kept separate from Stage 1's own audit script so Stage 1's
report still accurately reflects what was true when it was written.

Run: python model/train_v2.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from analysis.stage4_model_comparison import (
    NEW_SCHEMA_FEATURES,
    build_candidate_preprocessor,
    load_candidate_split,
)

MODEL_V2_PATH = BASE_DIR / "model" / "model_v2.pkl"
FEATURE_SCHEMA_PATH = BASE_DIR / "artifacts" / "feature_schema.json"
PREPROCESSING_METADATA_PATH = BASE_DIR / "artifacts" / "preprocessing_metadata.json"

MODEL_VERSION = "2.0.0"
SCHEMA_VERSION = "2.0.0"


def train_and_save_model() -> dict:
    X_train, X_test, y_train, y_test = load_candidate_split()
    preprocessor = build_candidate_preprocessor()
    classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_V2_PATH)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def promote_schema() -> None:
    with open(FEATURE_SCHEMA_PATH) as f:
        schema = json.load(f)

    schema["schema_version"] = SCHEMA_VERSION
    schema["production_features"] = NEW_SCHEMA_FEATURES
    schema["feature_order"] = NEW_SCHEMA_FEATURES
    schema["candidate_features"] = ["FarePerPerson", "Deck", "AgeGroup", "SibSp", "Parch", "IsAlone", "Sex_Pclass"]
    for feature in schema["features"]:
        if feature["name"] in NEW_SCHEMA_FEATURES:
            feature["production_status"] = "production"
        elif feature["name"] in ("SibSp", "Parch", "IsAlone"):
            feature["production_status"] = "candidate"
    schema["notes"] = (
        "Promoted to schema_version 2.0.0 by model/train_v2.py per the Stage 3 "
        "decision review and Stage 4/5 validation. SibSp/Parch remain required "
        "raw inputs (used to compute FamilySize) but are no longer direct model "
        "features; IsAlone was dropped as redundant with FamilySize."
    )
    with open(FEATURE_SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)

    with open(PREPROCESSING_METADATA_PATH) as f:
        metadata = json.load(f)
    metadata["model_version"] = MODEL_VERSION
    metadata["schema_version"] = SCHEMA_VERSION
    metadata["model_promoted_at"] = datetime.now(timezone.utc).isoformat()
    metadata["model_promotion_note"] = (
        "model/model_v2.pkl (Logistic Regression, 8-feature schema) promoted to "
        "production per Stage 4's controlled comparison and Stage 5's validation. "
        "model/model_v1.pkl (Random Forest, 11-feature schema) retained for rollback."
    )
    with open(PREPROCESSING_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    metrics = train_and_save_model()
    promote_schema()
    print(f"model_v2.pkl saved to {MODEL_V2_PATH}")
    print("Test-set metrics (should match Stage 4/5 exactly):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nartifacts/feature_schema.json -> schema_version {SCHEMA_VERSION}")
    print(f"artifacts/preprocessing_metadata.json -> model_version {MODEL_VERSION}")


if __name__ == "__main__":
    main()
