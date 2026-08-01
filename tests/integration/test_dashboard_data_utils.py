"""Integration: the dashboard's own data/model loading layer.

The dashboard does not call the API — it loads artifacts/preprocessing.pkl
and model/model_v2.pkl directly (see reports/stage6_deployment_plan.md's
architecture note). So the meaningful "API -> dashboard" integration check
today is: does the dashboard's loading layer actually produce correct,
usable objects — not a live HTTP call chain that doesn't exist yet.
"""
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DASHBOARD_DIR = BASE_DIR / "dashboard"
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import data_utils  # noqa: E402


def test_load_raw_returns_full_dataset():
    df = data_utils.load_raw.__wrapped__()
    assert len(df) == 891


def test_load_preprocessor_is_fitted():
    preprocessor = data_utils.load_preprocessor.__wrapped__()
    assert preprocessor.is_fitted_


def test_load_model_v2_predicts():
    model = data_utils.load_model_v2.__wrapped__()
    assert hasattr(model, "predict_proba")


def test_load_feature_schema_matches_artifact(base_dir):
    schema = data_utils.load_feature_schema.__wrapped__()
    assert schema["production_features"] == data_utils.PRODUCTION_FEATURES


def test_evaluate_v2_on_test_matches_stage4_report():
    """The dashboard's live-computed metrics should match Stage 4/5's
    already-reported numbers — same split, same model, same code path."""
    _, _, _, _, metrics = data_utils.evaluate_v2_on_test.__wrapped__()
    assert metrics["accuracy"] == pytest.approx(0.832, abs=0.001)
    assert metrics["f1"] == pytest.approx(0.792, abs=0.001)
    assert metrics["roc_auc"] == pytest.approx(0.865, abs=0.001)
