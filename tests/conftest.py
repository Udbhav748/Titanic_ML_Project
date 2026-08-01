import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


@pytest.fixture(scope="session")
def base_dir() -> Path:
    return BASE_DIR


@pytest.fixture(scope="session")
def raw_data() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "data" / "train.csv")


@pytest.fixture(scope="session")
def train_test_raw(raw_data):
    return train_test_split(raw_data, test_size=0.2, random_state=42, stratify=raw_data["Survived"])


@pytest.fixture(scope="session")
def fitted_preprocessor(train_test_raw):
    from src.preprocessing import TitanicPreprocessor

    train_raw, _ = train_test_raw
    return TitanicPreprocessor().fit(train_raw)


@pytest.fixture
def sample_passenger_complete() -> pd.DataFrame:
    """A fully-populated raw passenger row — no missing values."""
    return pd.DataFrame([{
        "PassengerId": 9001, "Pclass": 1, "Name": "Doe, Mrs. Jane",
        "Sex": "female", "Age": 29.0, "SibSp": 1, "Parch": 0,
        "Ticket": "PC 12345", "Fare": 80.0, "Cabin": "C85", "Embarked": "C",
    }])


@pytest.fixture
def sample_passenger_missing() -> pd.DataFrame:
    """A raw passenger row with the fields TitanicPreprocessor must impute."""
    return pd.DataFrame([{
        "PassengerId": 9002, "Pclass": 3, "Name": "Smith, Mr. John",
        "Sex": "male", "Age": None, "SibSp": 0, "Parch": 0,
        "Ticket": "A/5 21171", "Fare": 7.25, "Cabin": None, "Embarked": None,
    }])


@pytest.fixture
def sample_passenger_unseen_title() -> pd.DataFrame:
    """A title that will not appear in the fitted rare-title list."""
    return pd.DataFrame([{
        "PassengerId": 9003, "Pclass": 1, "Name": "Zorg, Cpt. Xylo",
        "Sex": "male", "Age": 40.0, "SibSp": 0, "Parch": 0,
        "Ticket": "X1", "Fare": 100.0, "Cabin": "B10", "Embarked": "S",
    }])
