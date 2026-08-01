"""Integration: preprocessing output feeding directly into model_v2 —
verifies the two components' contracts actually line up, not just that
each works in isolation."""
import joblib
import pytest

PRODUCTION_FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked", "HasCabin", "FamilySize", "Title"]


@pytest.fixture(scope="module")
def model_v2(base_dir):
    return joblib.load(base_dir / "model" / "model_v2.pkl")


def test_transformed_output_contains_all_production_features(fitted_preprocessor, sample_passenger_complete):
    transformed = fitted_preprocessor.transform(sample_passenger_complete)
    for feature in PRODUCTION_FEATURES:
        assert feature in transformed.columns


def test_model_accepts_preprocessor_output_directly(fitted_preprocessor, sample_passenger_complete, model_v2):
    transformed = fitted_preprocessor.transform(sample_passenger_complete)
    X = transformed[PRODUCTION_FEATURES]
    prediction = model_v2.predict(X)
    probability = model_v2.predict_proba(X)[:, 1]
    assert prediction[0] in (0, 1)
    assert 0.0 <= probability[0] <= 1.0


def test_model_handles_a_row_requiring_imputation(fitted_preprocessor, sample_passenger_missing, model_v2):
    transformed = fitted_preprocessor.transform(sample_passenger_missing)
    X = transformed[PRODUCTION_FEATURES]
    probability = model_v2.predict_proba(X)[:, 1]
    assert 0.0 <= probability[0] <= 1.0


def test_model_handles_unseen_title_and_deck(fitted_preprocessor, sample_passenger_unseen_title, model_v2):
    transformed = fitted_preprocessor.transform(sample_passenger_unseen_title)
    X = transformed[PRODUCTION_FEATURES]
    probability = model_v2.predict_proba(X)[:, 1]
    assert 0.0 <= probability[0] <= 1.0


def test_known_high_survival_profile_predicts_high_probability(fitted_preprocessor, model_v2):
    """1st-class woman with a cabin — should score well above the base rate (38%)."""
    import pandas as pd
    row = pd.DataFrame([{
        "PassengerId": 1, "Pclass": 1, "Name": "Rich, Mrs. Wealthy", "Sex": "female",
        "Age": 30.0, "SibSp": 0, "Parch": 0, "Ticket": "T1", "Fare": 150.0,
        "Cabin": "B10", "Embarked": "C",
    }])
    transformed = fitted_preprocessor.transform(row)
    probability = model_v2.predict_proba(transformed[PRODUCTION_FEATURES])[:, 1]
    assert probability[0] > 0.7


def test_known_low_survival_profile_predicts_low_probability(fitted_preprocessor, model_v2):
    """3rd-class man, no cabin, alone — should score well below the base rate."""
    import pandas as pd
    row = pd.DataFrame([{
        "PassengerId": 1, "Pclass": 3, "Name": "Poor, Mr. Unlucky", "Sex": "male",
        "Age": 30.0, "SibSp": 0, "Parch": 0, "Ticket": "T1", "Fare": 7.5,
        "Cabin": None, "Embarked": "S",
    }])
    transformed = fitted_preprocessor.transform(row)
    probability = model_v2.predict_proba(transformed[PRODUCTION_FEATURES])[:, 1]
    assert probability[0] < 0.3
