"""Integration: FastAPI request/response contract, using TestClient against
the real app (no mocks). Tests /predict — model_v2.pkl, the Logistic
Regression selected in Stage 4/5 and shown throughout the dashboard as the
production model, served through the 8-feature schema in
artifacts/feature_schema.json."""
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "pclass": 1, "sex": "female", "age": 29.0, "sibsp": 0, "parch": 0,
    "fare": 100.0, "embarked": "C", "has_cabin": True, "title": "Mrs",
}


def test_health_endpoint_reports_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"
    assert body["model_version"] == "2.0.0"
    assert body["schema_version"] == "2.0.0"


def test_predict_valid_payload_returns_200():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_schema():
    response = client.post("/predict", json=VALID_PAYLOAD).json()
    assert "survived" in response
    assert "survival_probability" in response
    assert isinstance(response["survived"], bool)
    assert 0.0 <= response["survival_probability"] <= 1.0
    assert response["model_version"] == "2.0.0"


@pytest.mark.parametrize("missing_field", ["pclass", "sex", "age", "sibsp", "parch", "fare"])
def test_predict_missing_required_field_returns_422(missing_field):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_sex_returns_422():
    payload = {**VALID_PAYLOAD, "sex": "unknown"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_pclass_returns_422():
    payload = {**VALID_PAYLOAD, "pclass": 4}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_defaults_apply_when_optional_fields_omitted():
    minimal_payload = {
        "pclass": 3, "sex": "male", "age": 25.0, "sibsp": 0, "parch": 0, "fare": 7.5,
    }
    response = client.post("/predict", json=minimal_payload)
    assert response.status_code == 200


def test_predict_high_survival_profile_scores_above_baseline():
    response = client.post("/predict", json=VALID_PAYLOAD).json()
    assert response["survival_probability"] > 0.384  # dataset base rate


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"
