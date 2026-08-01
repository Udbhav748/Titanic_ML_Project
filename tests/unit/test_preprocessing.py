"""Unit tests for TitanicPreprocessor — fit/transform contract, imputation,
and the robustness safeguards built in Stage 1."""
import pandas as pd
import pytest

from src.preprocessing import TitanicPreprocessor


class TestFit:
    def test_fit_returns_self(self, train_test_raw):
        train_raw, _ = train_test_raw
        result = TitanicPreprocessor().fit(train_raw)
        assert isinstance(result, TitanicPreprocessor)

    def test_fit_captures_age_medians_per_title(self, fitted_preprocessor):
        assert fitted_preprocessor.age_median_by_title_ is not None
        assert "Mr" in fitted_preprocessor.age_median_by_title_
        assert "Master" in fitted_preprocessor.age_median_by_title_
        # A "Master" (young boy) should have a far lower median age than "Mr"
        assert fitted_preprocessor.age_median_by_title_["Master"] < fitted_preprocessor.age_median_by_title_["Mr"]

    def test_fit_captures_embarked_mode(self, fitted_preprocessor):
        assert fitted_preprocessor.embarked_mode_ in {"C", "Q", "S"}

    def test_fit_captures_known_decks(self, fitted_preprocessor):
        assert fitted_preprocessor.known_decks_ is not None
        assert len(fitted_preprocessor.known_decks_) > 0

    def test_unfitted_transform_raises(self, sample_passenger_complete):
        with pytest.raises(RuntimeError):
            TitanicPreprocessor().transform(sample_passenger_complete)


class TestTransformImputation:
    def test_missing_age_is_imputed(self, fitted_preprocessor, sample_passenger_missing):
        result = fitted_preprocessor.transform(sample_passenger_missing)
        assert result["Age"].notna().all()

    def test_missing_embarked_is_imputed(self, fitted_preprocessor, sample_passenger_missing):
        result = fitted_preprocessor.transform(sample_passenger_missing)
        assert result["Embarked"].iloc[0] == fitted_preprocessor.embarked_mode_

    def test_age_imputed_by_title_not_global_median(self, fitted_preprocessor):
        """A 'Master' with missing age should get the Master median, not the Mr median."""
        row = pd.DataFrame([{
            "PassengerId": 1, "Pclass": 3, "Name": "Doe, Master. Tim", "Sex": "male",
            "Age": None, "SibSp": 1, "Parch": 1, "Ticket": "T1", "Fare": 20.0,
            "Cabin": None, "Embarked": "S",
        }])
        result = fitted_preprocessor.transform(row)
        assert result["Age"].iloc[0] == fitted_preprocessor.age_median_by_title_["Master"]

    def test_missing_cabin_yields_has_cabin_zero(self, fitted_preprocessor, sample_passenger_missing):
        result = fitted_preprocessor.transform(sample_passenger_missing)
        assert result["HasCabin"].iloc[0] == 0
        assert result["Deck"].iloc[0] == "U"

    def test_present_cabin_yields_has_cabin_one(self, fitted_preprocessor, sample_passenger_complete):
        result = fitted_preprocessor.transform(sample_passenger_complete)
        assert result["HasCabin"].iloc[0] == 1
        assert result["Deck"].iloc[0] == "C"


class TestFeatureEngineering:
    def test_family_size_is_sibsp_plus_parch_plus_one(self, fitted_preprocessor, sample_passenger_complete):
        result = fitted_preprocessor.transform(sample_passenger_complete)
        assert result["FamilySize"].iloc[0] == 1 + 0 + 1  # SibSp=1, Parch=0

    def test_is_alone_true_when_family_size_one(self, fitted_preprocessor):
        row = pd.DataFrame([{
            "PassengerId": 1, "Pclass": 3, "Name": "Solo, Mr. Traveler", "Sex": "male",
            "Age": 30.0, "SibSp": 0, "Parch": 0, "Ticket": "T1", "Fare": 10.0,
            "Cabin": None, "Embarked": "S",
        }])
        result = fitted_preprocessor.transform(row)
        assert result["IsAlone"].iloc[0] == 1

    def test_title_extracted_from_name(self, fitted_preprocessor, sample_passenger_complete):
        result = fitted_preprocessor.transform(sample_passenger_complete)
        assert result["Title"].iloc[0] == "Mrs"

    def test_fare_per_person_divides_by_family_size(self, fitted_preprocessor, sample_passenger_complete):
        result = fitted_preprocessor.transform(sample_passenger_complete)
        assert result["FarePerPerson"].iloc[0] == pytest.approx(80.0 / 2)

    def test_age_group_binning(self, fitted_preprocessor):
        child_row = pd.DataFrame([{
            "PassengerId": 1, "Pclass": 3, "Name": "Doe, Master. Kid", "Sex": "male",
            "Age": 5.0, "SibSp": 0, "Parch": 2, "Ticket": "T1", "Fare": 20.0,
            "Cabin": None, "Embarked": "S",
        }])
        result = fitted_preprocessor.transform(child_row)
        assert result["AgeGroup"].iloc[0] == "Child"


class TestRobustnessSafeguards:
    def test_unseen_title_collapses_to_rare(self, fitted_preprocessor, sample_passenger_unseen_title):
        result = fitted_preprocessor.transform(sample_passenger_unseen_title)
        assert result["Title"].iloc[0] == "Rare"

    def test_unseen_deck_letter_maps_to_other(self, fitted_preprocessor):
        """T is a real Titanic deck but rare enough it may not appear in every fit sample;
        this only assumes it's absent for the assertion to be meaningful."""
        row = pd.DataFrame([{
            "PassengerId": 1, "Pclass": 1, "Name": "Doe, Mr. Test", "Sex": "male",
            "Age": 30.0, "SibSp": 0, "Parch": 0, "Ticket": "T1", "Fare": 50.0,
            "Cabin": "Z99", "Embarked": "S",
        }])
        result = fitted_preprocessor.transform(row)
        assert result["Deck"].iloc[0] in {"Z", "Other"}
        if "Z" not in (fitted_preprocessor.known_decks_ or []):
            assert result["Deck"].iloc[0] == "Other"

    def test_zero_family_size_does_not_raise_divide_by_zero(self, fitted_preprocessor):
        """FamilySize can't be 0 through normal input (SibSp/Parch >= 0), but the
        divide-by-zero guard should still hold if it ever is."""
        row = pd.DataFrame([{
            "PassengerId": 1, "Pclass": 1, "Name": "Doe, Mr. Test", "Sex": "male",
            "Age": 30.0, "SibSp": -1, "Parch": 0, "Ticket": "T1", "Fare": 50.0,
            "Cabin": None, "Embarked": "S",
        }])
        result = fitted_preprocessor.transform(row)
        assert result["FamilySize"].iloc[0] == 0
        assert pd.notna(result["FarePerPerson"].iloc[0])

    def test_column_order_preserved(self, fitted_preprocessor, sample_passenger_complete):
        original_columns = list(sample_passenger_complete.columns)
        result = fitted_preprocessor.transform(sample_passenger_complete)
        assert list(result.columns[:len(original_columns)]) == original_columns

    def test_transform_does_not_mutate_input(self, fitted_preprocessor, sample_passenger_missing):
        before = sample_passenger_missing.copy(deep=True)
        fitted_preprocessor.transform(sample_passenger_missing)
        pd.testing.assert_frame_equal(sample_passenger_missing, before)


class TestPublicApiSurface:
    def test_fit_transform_matches_separate_fit_and_transform(self, train_test_raw):
        train_raw, _ = train_test_raw
        combined = TitanicPreprocessor().fit_transform(train_raw.copy())
        separate = TitanicPreprocessor().fit(train_raw.copy()).transform(train_raw.copy())
        pd.testing.assert_frame_equal(combined, separate)

    def test_to_metadata_dict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            TitanicPreprocessor().to_metadata_dict(fitted_on="test", n_train_rows=0)

    def test_to_metadata_dict_contains_fitted_values(self, fitted_preprocessor):
        metadata = fitted_preprocessor.to_metadata_dict(fitted_on="test split", n_train_rows=712)
        assert metadata["imputation"]["age_median_by_title"] == fitted_preprocessor.age_median_by_title_
        assert metadata["preprocessing_version"]


class TestDeterminism:
    def test_transform_is_deterministic(self, fitted_preprocessor, sample_passenger_complete):
        result1 = fitted_preprocessor.transform(sample_passenger_complete)
        result2 = fitted_preprocessor.transform(sample_passenger_complete)
        pd.testing.assert_frame_equal(result1, result2)

    def test_refitting_on_same_split_gives_same_values(self, train_test_raw):
        train_raw, _ = train_test_raw
        p1 = TitanicPreprocessor().fit(train_raw)
        p2 = TitanicPreprocessor().fit(train_raw)
        assert p1.age_median_by_title_ == p2.age_median_by_title_
        assert p1.embarked_mode_ == p2.embarked_mode_
        assert p1.rare_titles_ == p2.rare_titles_
