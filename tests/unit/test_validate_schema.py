"""Unit tests for TitanicPreprocessor.validate_schema() — the fail-fast
contract check Stage 1 built ahead of Stage 6's API integration."""
import pytest


def test_valid_data_passes(fitted_preprocessor, sample_passenger_complete):
    fitted_preprocessor.validate_schema(sample_passenger_complete)  # should not raise


def test_missing_required_column_raises(fitted_preprocessor, sample_passenger_complete):
    broken = sample_passenger_complete.drop(columns=["Age"])
    with pytest.raises(ValueError, match="missing required column"):
        fitted_preprocessor.validate_schema(broken)


def test_unexpected_column_raises(fitted_preprocessor, sample_passenger_complete):
    broken = sample_passenger_complete.copy()
    broken["NotAColumn"] = 1
    with pytest.raises(ValueError, match="unexpected column"):
        fitted_preprocessor.validate_schema(broken)


def test_wrong_dtype_raises(fitted_preprocessor, sample_passenger_complete):
    broken = sample_passenger_complete.copy()
    broken["Fare"] = "not a number"
    with pytest.raises(ValueError, match="dtype"):
        fitted_preprocessor.validate_schema(broken)


def test_null_in_non_nullable_column_raises(fitted_preprocessor, sample_passenger_complete):
    broken = sample_passenger_complete.copy()
    broken["Sex"] = None
    with pytest.raises(ValueError, match="non-null"):
        fitted_preprocessor.validate_schema(broken)


def test_null_age_is_allowed(fitted_preprocessor, sample_passenger_missing):
    fitted_preprocessor.validate_schema(sample_passenger_missing)  # Age/Cabin/Embarked are nullable


@pytest.mark.parametrize("optional_col", ["PassengerId", "Survived", "Ticket"])
def test_known_optional_columns_are_allowed(fitted_preprocessor, sample_passenger_complete, optional_col):
    with_optional = sample_passenger_complete.copy()
    with_optional[optional_col] = 0
    fitted_preprocessor.validate_schema(with_optional)  # should not raise


def test_error_message_lists_every_problem_at_once(fitted_preprocessor, sample_passenger_complete):
    broken = sample_passenger_complete.drop(columns=["Age", "Fare"])
    broken["Extra"] = 1
    with pytest.raises(ValueError) as exc_info:
        fitted_preprocessor.validate_schema(broken)
    message = str(exc_info.value)
    assert "Age" in message
    assert "Fare" in message
    assert "Extra" in message


def test_fit_calls_validate_schema(train_test_raw):
    """fit() should reject malformed data before computing any statistics."""
    from src.preprocessing import TitanicPreprocessor

    train_raw, _ = train_test_raw
    broken = train_raw.drop(columns=["Cabin"])
    with pytest.raises(ValueError):
        TitanicPreprocessor().fit(broken)
