"""End-to-end: raw input -> preprocessing -> prediction -> API response ->
dashboard, checked for consistency wherever the same model is actually
supposed to be involved.

Known, documented gap (not a bug): the live API (/predict) still serves
model_v1.pkl (Random Forest, 11 features); the dashboard serves model_v2.pkl
(Logistic Regression, 8 features) directly. Stage 6 planned the migration;
it has not been executed yet. So this suite does NOT assert that API and
dashboard predictions match today — it asserts that each path is internally
consistent and that the divergence is exactly what Stage 6/7 documented,
not a silent, undocumented mismatch.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "dashboard"))

from api.main import app  # noqa: E402

PRODUCTION_FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked", "HasCabin", "FamilySize", "Title"]

client = TestClient(app)

RAW_PASSENGER = {
    "PassengerId": 1, "Pclass": 1, "Name": "Doe, Mrs. Jane", "Sex": "female",
    "Age": 29.0, "SibSp": 0, "Parch": 0, "Ticket": "PC1", "Fare": 100.0,
    "Cabin": "C85", "Embarked": "C",
}

API_PAYLOAD = {
    "pclass": 1, "sex": "female", "age": 29.0, "sibsp": 0, "parch": 0,
    "fare": 100.0, "embarked": "C", "has_cabin": True, "title": "Mrs",
}


def test_preprocessing_to_dashboard_prediction_is_consistent(fitted_preprocessor):
    """Raw input -> TitanicPreprocessor -> model_v2 gives the same result
    whether computed directly or via the dashboard's cached loader path."""
    import data_utils

    raw_df = pd.DataFrame([RAW_PASSENGER])
    transformed = fitted_preprocessor.transform(raw_df)
    direct_model = joblib.load(BASE_DIR / "model" / "model_v2.pkl")
    direct_proba = direct_model.predict_proba(transformed[PRODUCTION_FEATURES])[0, 1]

    dashboard_preprocessor = data_utils.load_preprocessor.__wrapped__()
    dashboard_model = data_utils.load_model_v2.__wrapped__()
    dashboard_transformed = dashboard_preprocessor.transform(raw_df)
    dashboard_proba = dashboard_model.predict_proba(dashboard_transformed[PRODUCTION_FEATURES])[0, 1]

    assert direct_proba == pytest.approx(dashboard_proba)


def test_full_pipeline_is_repeatable(fitted_preprocessor):
    """Running the same raw passenger through the full pipeline twice must
    give bit-identical results — no hidden randomness anywhere in the path."""
    model = joblib.load(BASE_DIR / "model" / "model_v2.pkl")
    raw_df = pd.DataFrame([RAW_PASSENGER])

    run_1 = model.predict_proba(fitted_preprocessor.transform(raw_df)[PRODUCTION_FEATURES])[0, 1]
    run_2 = model.predict_proba(fitted_preprocessor.transform(raw_df)[PRODUCTION_FEATURES])[0, 1]

    assert run_1 == run_2


def test_api_response_is_well_formed_for_the_same_passenger():
    response = client.post("/predict", json=API_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["survival_probability"] <= 1.0


def test_api_v1_and_pipeline_v2_predictions_are_independently_valid_but_not_required_to_match(fitted_preprocessor):
    """Documents the known Stage 6 migration gap: v1 (API) and v2 (pipeline/
    dashboard) are different models by design until cutover. Both must be
    valid predictions; they are NOT asserted equal."""
    api_probability = client.post("/predict", json=API_PAYLOAD).json()["survival_probability"]

    raw_df = pd.DataFrame([RAW_PASSENGER])
    v2_model = joblib.load(BASE_DIR / "model" / "model_v2.pkl")
    v2_probability = v2_model.predict_proba(fitted_preprocessor.transform(raw_df)[PRODUCTION_FEATURES])[0, 1]

    assert 0.0 <= api_probability <= 1.0
    assert 0.0 <= v2_probability <= 1.0
    # Both scores should at least agree on the qualitative call for this
    # unambiguous high-survival profile (1st class, female, cabin recorded) —
    # a sanity check that v1 and v2 aren't wildly incompatible in judgment,
    # even though their exact probabilities are expected to differ.
    assert (api_probability > 0.5) == (v2_probability > 0.5)
