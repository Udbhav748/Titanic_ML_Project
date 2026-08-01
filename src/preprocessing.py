"""
Reusable, leakage-safe preprocessing for the Titanic feature set.

`TitanicPreprocessor` is fit exactly once, on a training split. Every
value it needs at inference time (per-Title Age medians, the Embarked
mode, the rare-title list, the set of Deck letters seen in training) is
captured in `fit()` and stored on the instance; `transform()` only ever
reads those stored values back — it never recomputes a median, mode, or
value_counts() from whatever data is passed to it. That's what makes it
safe to call on a single-row prediction request: there's no group in
one row to average over, so the fitted lookup table is the only thing
that can fill it correctly.

Serialize a fitted instance with joblib (see analysis/stage1_data_audit.py)
and load the same object at inference time instead of re-deriving these
values from request data.
"""
from __future__ import annotations

import pandas as pd

# Bump PREPROCESSING_VERSION whenever fit()/transform() logic changes in a
# way that could shift the values a previously-fitted artifact would have
# produced (new feature, changed imputation rule, changed bin edges, ...).
# It tracks the CODE; artifacts/preprocessing_metadata.json's
# creation_timestamp/git_commit track the specific FIT INSTANCE built from it.
PREPROCESSING_VERSION = "1.1.0"
FEATURE_SCHEMA_VERSION = "1.0.0"

RARE_TITLE_MIN_COUNT = 10
TITLE_NORMALIZE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
KNOWN_TITLE_BUCKETS = {"Mr", "Mrs", "Miss", "Master", "Rare"}
TITLE_REGEX = r",\s*([^\.]+)\."

DECK_UNKNOWN_LABEL = "U"   # Cabin missing entirely
DECK_OTHER_LABEL = "Other"  # Cabin present but its first letter was never seen at fit time

AGE_GROUP_BINS = [0, 12, 18, 59, 120]
AGE_GROUP_LABELS = ["Child", "Teen", "Adult", "Senior"]

# The engineered columns transform() adds, in the fixed order they should
# appear in the output (see _reorder_columns). Anything already present in
# the input (e.g. re-transforming already-transformed data) keeps its
# original position instead of being duplicated here.
ENGINEERED_COLUMN_ORDER = [
    "Title", "HasCabin", "FamilySize", "IsAlone", "FarePerPerson", "Deck", "AgeGroup",
]

# Raw input contract validate_schema() checks. Required = must be present;
# nullable = allowed to be missing (TitanicPreprocessor imputes it); anything
# outside REQUIRED + KNOWN_OPTIONAL is flagged as an unexpected column.
REQUIRED_RAW_COLUMNS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
NULLABLE_RAW_COLUMNS = {"Age", "Cabin", "Embarked"}
KNOWN_OPTIONAL_COLUMNS = {"PassengerId", "Survived", "Ticket"}
RAW_COLUMN_DTYPE_KINDS = {
    "Pclass": "iu", "Name": "O", "Sex": "O", "Age": "fiu", "SibSp": "iu",
    "Parch": "iu", "Fare": "fiu", "Cabin": "O", "Embarked": "O",
}


class TitanicPreprocessor:
    """Fit on a raw training DataFrame; transform raw rows (train, test, or
    a single inference request) using only the values captured at fit time.
    """

    def __init__(self) -> None:
        self.rare_titles_: list[str] | None = None
        self.age_median_by_title_: dict[str, float] | None = None
        self.age_median_overall_: float | None = None
        self.embarked_mode_: str | None = None
        self.known_decks_: list[str] | None = None
        self.is_fitted_: bool = False

    @staticmethod
    def _extract_raw_title(data: pd.DataFrame) -> pd.Series:
        return data["Name"].str.extract(TITLE_REGEX)[0]

    def _normalize_title(self, raw_title: pd.Series) -> pd.Series:
        title = raw_title.replace(self.rare_titles_, "Rare")
        title = title.replace(TITLE_NORMALIZE_MAP)
        # Inference safety net: a title string never seen at fit time (and
        # not already one of the five known buckets) collapses to "Rare"
        # instead of leaking an unseen category into the Age lookup.
        title = title.where(title.isin(KNOWN_TITLE_BUCKETS), "Rare")
        return title

    def validate_schema(self, data: pd.DataFrame) -> None:
        """
        Check `data` against the raw input contract before it ever reaches
        fit()/transform() internals, and raise one informative ValueError
        listing everything wrong at once — instead of letting a missing
        column or wrong dtype surface as a confusing KeyError/TypeError
        deep inside a fillna or string accessor. Intended to be called by
        FastAPI request handling in Stage 6 as well as internally here.
        """
        errors: list[str] = []
        columns = set(data.columns)

        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in columns]
        if missing:
            errors.append(f"missing required column(s): {missing}")

        unexpected = sorted(columns - set(REQUIRED_RAW_COLUMNS) - KNOWN_OPTIONAL_COLUMNS)
        if unexpected:
            errors.append(f"unexpected column(s) not in the known schema: {unexpected}")

        for col in REQUIRED_RAW_COLUMNS:
            if col not in columns:
                continue
            series = data[col]
            if col not in NULLABLE_RAW_COLUMNS and series.isna().any():
                errors.append(
                    f"column '{col}' is required to be non-null but contains "
                    f"{int(series.isna().sum())} null value(s)"
                )
            expected_kinds = RAW_COLUMN_DTYPE_KINDS[col]
            if series.dropna().shape[0] > 0 and series.dtype.kind not in expected_kinds:
                errors.append(
                    f"column '{col}' has dtype '{series.dtype}' (kind='{series.dtype.kind}'), "
                    f"expected kind in '{expected_kinds}'"
                )

        if errors:
            raise ValueError(
                "TitanicPreprocessor.validate_schema failed:\n- " + "\n- ".join(errors)
            )

    def fit(self, data: pd.DataFrame) -> "TitanicPreprocessor":
        """Fit on the TRAINING SPLIT ONLY. Never call this on test/inference data."""
        self.validate_schema(data)
        data = data.copy()
        raw_title = self._extract_raw_title(data)

        title_counts = raw_title.value_counts()
        self.rare_titles_ = title_counts[title_counts < RARE_TITLE_MIN_COUNT].index.tolist()

        title = self._normalize_title(raw_title)
        age_by_title = data.assign(_Title=title).groupby("_Title")["Age"].median()
        self.age_median_by_title_ = age_by_title.round(2).to_dict()
        self.age_median_overall_ = round(float(data["Age"].median()), 2)

        self.embarked_mode_ = data["Embarked"].mode(dropna=True).iloc[0]

        self.known_decks_ = sorted(data["Cabin"].dropna().astype(str).str[0].unique().tolist())

        self.is_fitted_ = True
        return self

    def _reorder_columns(self, data: pd.DataFrame, original_columns: list[str]) -> pd.DataFrame:
        """Guarantee deterministic output column order: every input column
        keeps its original position, engineered columns are appended in a
        fixed order. Re-transforming already-transformed data (which already
        has e.g. 'Title') doesn't duplicate those columns."""
        appended = [c for c in ENGINEERED_COLUMN_ORDER if c not in original_columns]
        return data[original_columns + appended]

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation + feature engineering. Fit-only values are
        read back from `self.*_`, never recomputed from `data`."""
        if not self.is_fitted_:
            raise RuntimeError("TitanicPreprocessor.transform() called before fit().")
        self.validate_schema(data)

        original_columns = list(data.columns)
        data = data.copy()  # never mutate the caller's DataFrame / avoid chained-assignment warnings

        raw_title = self._extract_raw_title(data)
        data["Title"] = self._normalize_title(raw_title)

        # Coerce to float64 before filling: a single-row DataFrame built from a
        # dict with Age=None otherwise arrives as object dtype, which makes
        # pandas' fillna emit a downcasting FutureWarning on the numeric fill.
        data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
        age_lookup = data["Title"].map(self.age_median_by_title_)
        data["Age"] = data["Age"].fillna(age_lookup).fillna(self.age_median_overall_)

        data["Embarked"] = data["Embarked"].fillna(self.embarked_mode_)

        data["HasCabin"] = data["Cabin"].notna().astype(int)

        data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
        data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

        # FamilySize is mathematically >= 1 given non-negative SibSp/Parch
        # (enforced by the API's Pydantic schema), but this preprocessor may
        # also be called directly (e.g. from analysis scripts) without that
        # upstream guarantee, so the divide is guarded defensively anyway.
        family_size_safe = data["FamilySize"].where(data["FamilySize"] != 0, 1)
        data["FarePerPerson"] = data["Fare"] / family_size_safe

        raw_deck = data["Cabin"].str[0]
        known_decks = set(self.known_decks_)
        # Cabin present but its first letter was never seen at fit time ->
        # "Other", distinct from "U" (no cabin recorded at all).
        deck = raw_deck.where(raw_deck.isna() | raw_deck.isin(known_decks), DECK_OTHER_LABEL)
        data["Deck"] = deck.fillna(DECK_UNKNOWN_LABEL)

        data["AgeGroup"] = pd.cut(
            data["Age"], bins=AGE_GROUP_BINS, labels=AGE_GROUP_LABELS, right=True
        )

        return self._reorder_columns(data, original_columns)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    def to_metadata_dict(self, fitted_on: str, n_train_rows: int) -> dict:
        """Human-readable mirror of the fitted values, for
        artifacts/preprocessing_metadata.json."""
        if not self.is_fitted_:
            raise RuntimeError("to_metadata_dict() called before fit().")
        return {
            "fitted_on": fitted_on,
            "n_train_rows": n_train_rows,
            "imputation": {
                "age_median_by_title": self.age_median_by_title_,
                "age_median_overall_fallback": self.age_median_overall_,
                "embarked_mode": self.embarked_mode_,
                "rare_titles_collapsed": self.rare_titles_,
                "known_decks": self.known_decks_,
            },
            "static_config": {
                "rare_title_min_count": RARE_TITLE_MIN_COUNT,
                "title_normalize_map": TITLE_NORMALIZE_MAP,
                "title_normalize_map_note": (
                    "Mlle/Ms/Mme all occur fewer than rare_title_min_count times in "
                    "this dataset, so the rare-title collapse step absorbs them into "
                    "'Rare' before this map would ever run. Kept for parity with "
                    "model/train.py's existing logic; documented as a known no-op "
                    "in reports/stage1_summary.md rather than silently changed."
                ),
                "unseen_title_fallback": "Rare",
                "deck_unknown_label": DECK_UNKNOWN_LABEL,
                "deck_other_label": DECK_OTHER_LABEL,
                "age_group_bins": AGE_GROUP_BINS,
                "age_group_labels": AGE_GROUP_LABELS,
            },
            "preprocessing_class": "src.preprocessing.TitanicPreprocessor",
            "preprocessing_version": PREPROCESSING_VERSION,
        }
