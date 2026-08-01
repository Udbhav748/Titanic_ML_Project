"""Unit tests for the versioned artifacts themselves — loadability and
cross-file version consistency.

test_schema_version_matches_across_artifacts is a regression test for a
real bug: Stage 7's promotion script updated feature_schema.json's
schema_version but not preprocessing_metadata.json's own copy of the same
field, leaving the two artifacts silently disagreeing. This test exists so
that specific mistake can never ship again unnoticed.
"""
import json

import joblib
import pytest


def test_preprocessing_pkl_loads(base_dir):
    preprocessor = joblib.load(base_dir / "artifacts" / "preprocessing.pkl")
    assert preprocessor.is_fitted_


def test_model_v2_pkl_loads(base_dir):
    model = joblib.load(base_dir / "model" / "model_v2.pkl")
    assert hasattr(model, "predict_proba")


def test_model_v1_pkl_loads(base_dir):
    model = joblib.load(base_dir / "model" / "model_v1.pkl")
    assert hasattr(model, "predict_proba")


def test_feature_schema_json_has_required_keys(base_dir):
    with open(base_dir / "artifacts" / "feature_schema.json") as f:
        schema = json.load(f)
    for key in ("schema_version", "production_features", "candidate_features",
                "engineered_features", "feature_order", "features"):
        assert key in schema


def test_preprocessing_metadata_json_has_required_keys(base_dir):
    with open(base_dir / "artifacts" / "preprocessing_metadata.json") as f:
        metadata = json.load(f)
    for key in ("preprocessing_version", "model_version", "schema_version",
                "sklearn_version", "python_version"):
        assert key in metadata


def test_schema_version_matches_across_artifacts(base_dir):
    """Regression test for the Stage 7 version-drift bug."""
    with open(base_dir / "artifacts" / "feature_schema.json") as f:
        schema = json.load(f)
    with open(base_dir / "artifacts" / "preprocessing_metadata.json") as f:
        metadata = json.load(f)
    assert schema["schema_version"] == metadata["schema_version"], (
        "artifacts/feature_schema.json and artifacts/preprocessing_metadata.json "
        "disagree on schema_version — this is exactly the drift bug found in Stage 7."
    )


def test_model_version_is_populated_once_promoted(base_dir):
    with open(base_dir / "artifacts" / "preprocessing_metadata.json") as f:
        metadata = json.load(f)
    assert metadata["model_version"] is not None


def test_production_features_match_model_v2_expected_columns(base_dir):
    with open(base_dir / "artifacts" / "feature_schema.json") as f:
        schema = json.load(f)
    model = joblib.load(base_dir / "model" / "model_v2.pkl")
    expected_columns = set()
    for _, _, cols in model.named_steps["preprocessor"].transformers:
        expected_columns.update(cols)
    assert expected_columns == set(schema["production_features"])


@pytest.mark.parametrize("status", ["production", "candidate", "deprecated"])
def test_every_feature_has_a_valid_production_status(base_dir, status):
    with open(base_dir / "artifacts" / "feature_schema.json") as f:
        schema = json.load(f)
    statuses = {f["production_status"] for f in schema["features"]}
    assert statuses <= {"production", "candidate", "deprecated"}
