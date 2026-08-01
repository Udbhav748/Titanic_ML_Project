"""
Stage 1 — Data Understanding & Cleaning.

Fits TitanicPreprocessor on the train split, audits every feature against
Survived (chi-square / point-biserial), and writes the reports + artifacts
in reports/ and artifacts/.

Run: python analysis/stage1_data_audit.py
"""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import (  # noqa: E402
    AGE_GROUP_LABELS,
    FEATURE_SCHEMA_VERSION,
    PREPROCESSING_VERSION,
    TitanicPreprocessor,
)

DATA_PATH = BASE_DIR / "data" / "train.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
PREPROCESSING_PKL_PATH = ARTIFACTS_DIR / "preprocessing.pkl"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
PREPROCESSING_METADATA_PATH = ARTIFACTS_DIR / "preprocessing_metadata.json"

REPORTS_DIR = BASE_DIR / "reports"
AUDIT_REPORT_PATH = REPORTS_DIR / "stage1_data_audit.md"
DATA_DICTIONARY_PATH = REPORTS_DIR / "data_dictionary.md"
SUMMARY_PATH = REPORTS_DIR / "stage1_summary.md"

PROD_SCHEMA_FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
    "HasCabin", "FamilySize", "IsAlone", "Title",
]
CANDIDATE_FEATURES = ["FarePerPerson", "Deck", "AgeGroup"]
ENGINEERED_FEATURES = ["HasCabin", "FamilySize", "IsAlone", "Title", "FarePerPerson", "Deck", "AgeGroup"]

CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "HasCabin", "IsAlone", "Title", "Deck", "AgeGroup"]
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "FarePerPerson"]
COUNT_FEATURES_NEEDING_NONLINEAR_CHECK = {"SibSp", "Parch", "FamilySize"}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def report_shape_and_missingness(data: pd.DataFrame) -> str:
    lines = ["## Data Overview\n"]
    lines.append(f"- Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    lines.append(f"- Duplicate rows: {data.duplicated().sum()}")
    lines.append(f"- Duplicate `PassengerId` values: {data['PassengerId'].duplicated().sum()}\n")

    dtypes_df = data.dtypes.rename("dtype").to_frame()
    missing = data.isna().sum().rename("missing_count")
    missing_pct = (data.isna().mean() * 100).round(2).rename("missing_pct")
    summary = pd.concat([dtypes_df, missing, missing_pct], axis=1)

    lines.append("| Column | dtype | missing_count | missing_pct |")
    lines.append("|---|---|---|---|")
    for col, row in summary.iterrows():
        lines.append(f"| {col} | {row['dtype']} | {int(row['missing_count'])} | {row['missing_pct']}% |")

    return "\n".join(lines)


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    r, c = confusion_matrix.shape
    return float(np.sqrt((chi2 / n) / (min(r, c) - 1)))


def audit_categorical(data: pd.DataFrame, feature: str, target: str = "Survived") -> dict:
    table = pd.crosstab(data[feature], data[target])
    chi2, p, dof, expected = chi2_contingency(table)
    return {
        "feature": feature,
        "test": "chi-square",
        "statistic": round(float(chi2), 3),
        "p_value": p,
        "effect_size": round(cramers_v(table), 3),
        "effect_label": "Cramer's V",
        "min_expected_freq_below_5": bool((expected < 5).any()),
    }


def audit_numeric(data: pd.DataFrame, feature: str, target: str = "Survived") -> dict:
    valid = data[[feature, target]].dropna()
    r, p = pointbiserialr(valid[target], valid[feature])
    return {
        "feature": feature,
        "test": "point-biserial r (≡ two-group ANOVA/t-test)",
        "statistic": round(float(r), 3),
        "p_value": p,
        "effect_size": round(abs(float(r)), 3),
        "effect_label": "|point-biserial r|",
        "min_expected_freq_below_5": None,
    }


def nonlinear_recheck_p(data: pd.DataFrame, feature: str, target: str = "Survived") -> float:
    table = pd.crosstab(data[feature], data[target])
    return float(chi2_contingency(table)[1])


def decide(p_value: float, caveat: str | None, nonlinear_p: float | None) -> tuple[str, str | None]:
    """Return (decision, extra_caveat). extra_caveat is auto-generated only
    when the linear/categorical disagreement isn't already explained by a
    hand-written caveat (avoids saying the same thing twice in the table)."""
    if p_value >= 0.05:
        if caveat:
            return "keep-with-caveat", None
        if nonlinear_p is not None and nonlinear_p < 0.05:
            extra = (
                f"Linear test alone is NOT significant (p={p_value:.3f}), but a chi-square "
                f"treating it as categorical IS (p={nonlinear_p:.4f}) — the true relationship "
                f"is non-monotonic, so point-biserial correlation understates it. Keeping on "
                f"that basis, not the linear test."
            )
            return "keep-with-caveat", extra
        return "drop", None
    if caveat:
        return "keep-with-caveat", None
    return "keep", None


def build_audit_table(full: pd.DataFrame) -> pd.DataFrame:
    caveats = {
        "HasCabin": "Strongly overlaps with Deck (Deck='U' <=> HasCabin=0); keep HasCabin, treat Deck as exploratory only.",
        "Deck": "77% of rows fall in the single 'U' (unknown) bucket and several real decks have <5 expected count per cell — chi-square assumptions are shaky. Redundant with HasCabin.",
        "FamilySize": "Point-biserial correlation is essentially zero (p=0.62) because the true relationship is non-monotonic — mid-size families (2-4) survived best, solo travelers and large families worst — which a linear test can't see. A chi-square treating it as categorical IS significant (p<0.001). Keep-with-caveat: real signal, wrong linear test.",
        "IsAlone": "Deterministic function of FamilySize (IsAlone = FamilySize==1); redundant signal, keep only if it linearly separates better than FamilySize alone.",
        "AgeGroup": "Derived from Age via fixed bins; binning trades continuous signal for interpretability. Compare against raw Age importance in Stage 2.",
        "FarePerPerson": "Derived from Fare / FamilySize; check VIF against both parents in Stage 2 before finalizing.",
        "SibSp": "Same story as FamilySize: not significant as a linear predictor (p=0.29) but is significant as a categorical one (p<0.001) — non-monotonic, mostly redundant with FamilySize. Keep-with-caveat.",
        "Parch": "Weak on its own (p=0.015, small effect); contributes mainly through FamilySize. Keep-with-caveat.",
    }

    rows = [audit_categorical(full, f) for f in CATEGORICAL_FEATURES]
    rows += [audit_numeric(full, f) for f in NUMERIC_FEATURES]

    audit_df = pd.DataFrame(rows)
    audit_df["in_prod_schema"] = audit_df["feature"].isin(PROD_SCHEMA_FEATURES)

    decisions, extra_caveats = [], []
    for _, r in audit_df.iterrows():
        nonlinear_p = (
            nonlinear_recheck_p(full, r["feature"])
            if r["feature"] in COUNT_FEATURES_NEEDING_NONLINEAR_CHECK else None
        )
        decision, extra = decide(r["p_value"], caveats.get(r["feature"]), nonlinear_p)
        decisions.append(decision)
        extra_caveats.append(extra)
    audit_df["decision"] = decisions
    audit_df["caveat"] = [
        (caveats.get(f, "") + (" " + e if e else "")).strip()
        for f, e in zip(audit_df["feature"], extra_caveats)
    ]
    return audit_df.sort_values("effect_size", ascending=False).reset_index(drop=True)


def build_leakage_section(embarked_mode: str) -> str:
    return f"""## Leakage Prevention

Split happens before anything is fit:

1. `train_test_split` (80/20, stratified, `random_state=42`)
2. `TitanicPreprocessor.fit(train_raw)` — medians, mode, rare-title list computed from the training split only
3. `transform()` reads those fitted values back, never recomputes stats from its input
4. Fitted preprocessor serialized to `artifacts/preprocessing.pkl`
5. Inference loads it once at startup, calls `.transform()` only — never `.fit()`

Recomputing a median at inference time would quietly shift the model's inputs away from what it was trained on. A single request has no group to average over anyway.

**Found while building this:** `model/train.py` fits its Age-by-Title medians on the full dataset, before splitting — a small leak. `TitanicPreprocessor` fixes it (embarked mode fit on train split: `{embarked_mode}`); `model/model.pkl` doesn't have the fix until it's retrained.

**Takeaway**

Fit on train only, persist the values, never refit at inference. `model/train.py` needs a retrain to close the gap.
"""


def build_deck_hascabin_section() -> str:
    return """## Deck vs. HasCabin

`Deck` is just `Cabin[0]`, with missing Cabin mapped to `'U'`. So `HasCabin=0` and `Deck='U'` carry the same information, and several deck letters have fewer than 10 rows each.

Keeping both risks:
- redundant one-hot columns, ~90% collinear with `HasCabin`
- unstable coefficients/splits from near-duplicate features

Which one survives depends on VIF and feature importance (Stage 2), not a guess made here.

**Takeaway**

Deck and HasCabin overlap almost completely. Final call deferred to Stage 3.
"""


def build_familysize_section() -> str:
    return """## FamilySize: Linear Test Missed It

Point-biserial correlation is ~0 (r=0.017, p=0.62) — looks like no effect.

But survival by family size isn't linear: solo travelers do worse, families of 2-4 do best, large families do worst. A linear test averages the rise and the fall to zero.

A chi-square treating FamilySize as categorical picks it up clearly (p<0.001).

**Takeaway**

FamilySize carries real signal — the linear test was the wrong tool. Tree models will capture this shape natively; logistic regression won't unless it's binned.
"""


def render_audit_table_markdown(audit_df: pd.DataFrame) -> str:
    lines = [
        "## Statistical Audit\n",
        "Categorical → chi-square (Cramer's V). Numeric → point-biserial correlation "
        "(≡ ANOVA for 2 groups). Sorted by effect size.\n",
        "| Feature | In prod schema | Test | Statistic | p-value | Effect size | Decision | Caveat |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in audit_df.iterrows():
        p_str = "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.4f}"
        lines.append(
            f"| {r['feature']} | {'yes' if r['in_prod_schema'] else 'no'} | {r['test']} | "
            f"{r['statistic']} | {p_str} | {r['effect_size']} ({r['effect_label']}) | "
            f"**{r['decision']}** | {r['caveat']} |"
        )
    return "\n".join(lines)


VERDICT_NOTES = {
    "Title": "Encodes sex + age + status in one feature. Strongest categorical effect. Expect VIF overlap with Sex/Age in Stage 2.",
    "Sex": "Largest effect in the dataset. Women survived ~74% vs. ~19% for men.",
    "Pclass": "Strong effect — proxy for socioeconomic status and lifeboat access.",
    "Deck": "Significant mostly because 'U' dominates. See Deck vs. HasCabin below.",
    "HasCabin": "Solid, low-noise binary signal despite 77% missingness.",
    "Fare": "Strong effect, right-skewed. Log-transform candidate for Stage 2.",
    "FarePerPerson": "Cleaner separation than raw Fare, but correlated with both parents. VIF decides in Stage 2.",
    "IsAlone": "Deterministic threshold on FamilySize — largely redundant. Compare importance in Stage 2.",
    "Embarked": "Significant, but likely a Pclass proxy (README Phase 1). Recheck conditional on Pclass in Stage 2.",
    "AgeGroup": "Effect driven almost entirely by the Child bucket. Raw Age likely carries more information.",
    "Parch": "Weak on its own (p=0.015, small effect). Contributes mainly through FamilySize.",
    "Age": "Modest linear effect. Young children survive notably better.",
    "SibSp": "Same story as FamilySize — not linear, but significant as categorical. See FamilySize section below.",
    "FamilySize": "Linear correlation is ~0 because the relationship is non-monotonic. See section below.",
}


def render_verdicts_markdown(audit_df: pd.DataFrame) -> str:
    lines = ["## Per-Feature Verdicts\n"]
    for feature in audit_df["feature"]:
        if feature in VERDICT_NOTES:
            lines.append(f"- **{feature}** — {VERDICT_NOTES[feature]}")
    lines.append("")
    lines.append("**Takeaway**\n")
    lines.append(
        "Sex, Title, and Pclass dominate. Everything else is kept with a caveat "
        "pending Stage 2/3 evidence — nothing is dropped on Stage 1 alone."
    )
    return "\n".join(lines)


# Single source of truth for both artifacts/feature_schema.json and
# reports/data_dictionary.md, so the two can never drift apart. Excludes
# `Survived` (the target label, not an input feature) — that gets a
# hand-written row in build_data_dictionary() instead.
FEATURE_CATALOG = [
    {
        "name": "PassengerId", "type_label": "Numeric (identifier)", "dtype": "int64",
        "source": "original", "source_detail": "Original dataset",
        "required": False, "nullable": False, "default": None,
        "description": "Unique row identifier; no predictive signal.",
        "production_status": "deprecated",
    },
    {
        "name": "Pclass", "type_label": "Categorical (ordinal)", "dtype": "int64",
        "source": "original", "source_detail": "Original dataset",
        "required": True, "nullable": False, "default": None,
        "description": "Ticket class (1/2/3); proxy for socioeconomic status and physical deck location.",
        "production_status": "production",
    },
    {
        "name": "Name", "type_label": "Text", "dtype": "string",
        "source": "original", "source_detail": "Original dataset",
        "required": True, "nullable": False, "default": None,
        "description": "Free-text passenger name; not used directly, but the source column for Title extraction.",
        "production_status": "deprecated",
    },
    {
        "name": "Sex", "type_label": "Categorical", "dtype": "string (male|female)",
        "source": "original", "source_detail": "Original dataset",
        "required": True, "nullable": False, "default": None,
        "description": "Passenger sex.",
        "production_status": "production",
    },
    {
        "name": "Age", "type_label": "Numeric", "dtype": "float64",
        "source": "original", "source_detail": "Original dataset (imputed)",
        "required": True, "nullable": True, "default": None,
        "description": "Age in years. Nullable on input; TitanicPreprocessor fills missing values with the fitted per-Title training median, never a static default.",
        "production_status": "production",
    },
    {
        "name": "SibSp", "type_label": "Numeric (count)", "dtype": "int64",
        "source": "original", "source_detail": "Original dataset",
        "required": True, "nullable": False, "default": None,
        "description": "Siblings/spouses aboard.",
        "production_status": "production",
    },
    {
        "name": "Parch", "type_label": "Numeric (count)", "dtype": "int64",
        "source": "original", "source_detail": "Original dataset",
        "required": True, "nullable": False, "default": None,
        "description": "Parents/children aboard.",
        "production_status": "production",
    },
    {
        "name": "Ticket", "type_label": "Text", "dtype": "string",
        "source": "original", "source_detail": "Original dataset",
        "required": False, "nullable": False, "default": None,
        "description": "Ticket number; high-cardinality identifier with no clean signal.",
        "production_status": "deprecated",
    },
    {
        "name": "Fare", "type_label": "Numeric", "dtype": "float64",
        "source": "original", "source_detail": "Original dataset",
        "required": True, "nullable": False, "default": None,
        "description": "Fare paid; right-skewed (flagged for a log-transform check in Stage 2).",
        "production_status": "production",
    },
    {
        "name": "Cabin", "type_label": "Text", "dtype": "string",
        "source": "original", "source_detail": "Original dataset",
        "required": False, "nullable": True, "default": None,
        "description": "77% missing. Not used directly — decomposed into HasCabin and Deck instead.",
        "production_status": "deprecated",
    },
    {
        "name": "Embarked", "type_label": "Categorical", "dtype": "string (C|Q|S)",
        "source": "original", "source_detail": "Original dataset (imputed)",
        "required": False, "nullable": True, "default": "S",
        "description": "Port of embarkation. Nullable on input; TitanicPreprocessor fills missing values with the fitted training-split mode (currently 'S').",
        "production_status": "production",
    },
    {
        "name": "HasCabin", "type_label": "Binary", "dtype": "int64 (0|1)",
        "source": "engineered", "source_detail": "Engineered (from Cabin)",
        "required": False, "nullable": False, "default": False,
        "description": "1 if a cabin was recorded, 0 otherwise.",
        "production_status": "production",
    },
    {
        "name": "FamilySize", "type_label": "Numeric (count)", "dtype": "int64",
        "source": "engineered", "source_detail": "Engineered (SibSp + Parch + 1)",
        "required": False, "nullable": False, "default": None,
        "description": "Total family members aboard including the passenger; non-monotonic effect on survival (see reports/stage1_data_audit.md).",
        "production_status": "production",
    },
    {
        "name": "IsAlone", "type_label": "Binary", "dtype": "int64 (0|1)",
        "source": "engineered", "source_detail": "Engineered (FamilySize == 1)",
        "required": False, "nullable": False, "default": None,
        "description": "1 if traveling without any family aboard. Deterministic function of FamilySize.",
        "production_status": "production",
    },
    {
        "name": "Title", "type_label": "Categorical", "dtype": "string (Mr|Mrs|Miss|Master|Rare)",
        "source": "engineered", "source_detail": "Engineered (from Name)",
        "required": False, "nullable": False, "default": "Mr",
        "description": "Honorific extracted from Name; titles occurring <10 times in the training split (or never seen at fit time) collapse to 'Rare'.",
        "production_status": "production",
    },
    {
        "name": "FarePerPerson", "type_label": "Numeric", "dtype": "float64",
        "source": "engineered", "source_detail": "Engineered (Fare / FamilySize)",
        "required": False, "nullable": False, "default": None,
        "description": "Estimated per-individual fare, adjusting for group ticket pricing. Divide-by-zero guarded.",
        "production_status": "candidate",
    },
    {
        "name": "Deck", "type_label": "Categorical", "dtype": "string (A-G|T|Other|U)",
        "source": "engineered", "source_detail": "Engineered (from Cabin)",
        "required": False, "nullable": False, "default": "U",
        "description": "First letter of Cabin; 'U' where Cabin is missing, 'Other' where an unseen deck letter appears at inference. Highly redundant with HasCabin.",
        "production_status": "candidate",
    },
    {
        "name": "AgeGroup", "type_label": "Categorical", "dtype": "string (Child|Teen|Adult|Senior)",
        "source": "engineered", "source_detail": "Engineered (binned Age)",
        "required": False, "nullable": False, "default": None,
        "description": "Age binned into Child (<=12) / Teen (13-18) / Adult (19-59) / Senior (60+).",
        "production_status": "candidate",
    },
]


def build_feature_schema() -> dict:
    features = [
        {
            "name": f["name"],
            "dtype": f["dtype"],
            "source": f["source"],
            "required": f["required"],
            "nullable": f["nullable"],
            "default": f["default"],
            "description": f["description"],
            "production_status": f["production_status"],
        }
        for f in FEATURE_CATALOG
    ]
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "features": features,
        "production_features": list(PROD_SCHEMA_FEATURES),
        "candidate_features": CANDIDATE_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "feature_order": list(PROD_SCHEMA_FEATURES),
        "notes": (
            "This is intended as the single source of truth for the raw-input "
            "feature contract: FastAPI request validation, Streamlit form fields, "
            "and retraining scripts should all read from this file rather than "
            "hardcoding feature lists. candidate_features are not yet part of the "
            "served model's input contract (production_status='candidate'); see "
            "reports/stage1_data_audit.md for the per-feature keep/drop/"
            "keep-with-caveat rationale. The final production schema is decided in "
            "Stage 3 and should bump schema_version when it changes."
        ),
    }


def get_git_commit() -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, stderr=subprocess.DEVNULL,
        )
        return commit.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def write_artifacts(preprocessor: TitanicPreprocessor, n_train_rows: int) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, PREPROCESSING_PKL_PATH)

    metadata = preprocessor.to_metadata_dict(
        fitted_on="data/train.csv, 80% stratified split (test_size=0.2, random_state=42)",
        n_train_rows=n_train_rows,
    )

    # --- Artifact versioning (see "Why versioning matters" below) ---
    metadata["model_version"] = None  # model/model.pkl has no formal versioning scheme yet — Stage 4/6 will populate this
    metadata["schema_version"] = FEATURE_SCHEMA_VERSION
    metadata["creation_timestamp"] = datetime.now(timezone.utc).isoformat()
    metadata["training_dataset"] = "data/train.csv"
    metadata["sklearn_version"] = sklearn.__version__
    metadata["python_version"] = platform.python_version()
    metadata["git_commit"] = get_git_commit()
    metadata["artifact_path"] = str(PREPROCESSING_PKL_PATH.relative_to(BASE_DIR)).replace("\\", "/")
    metadata["versioning_rationale"] = (
        "Versioning preprocessing artifacts separately from the code that produced "
        "them makes it possible to answer 'exactly which imputation values, which "
        "feature schema, and which git commit produced the model currently running "
        "in production?' without re-deriving it. Without these fields, a production "
        "incident (unexpected predictions, an API/model mismatch after a deploy) has "
        "no fast way to confirm whether the model, the preprocessing artifact, or a "
        "code change was responsible — each field here (preprocessing_version, "
        "schema_version, model_version, git_commit, creation_timestamp) is a "
        "coordinate that lets you pin down exactly what was deployed and roll back "
        "precisely instead of guessing. preprocessing_version tracks the *code* "
        "(bump it when fit()/transform() logic changes in a way that could shift "
        "fitted values); creation_timestamp/git_commit/training_dataset track this "
        "specific *fitted instance's* provenance. model_version is null here because "
        "model/model.pkl has no formal versioning scheme yet — the Stage 4 retrain "
        "is the natural point to introduce one and populate this field."
    )

    with open(PREPROCESSING_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    schema = build_feature_schema()
    with open(FEATURE_SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)

    return metadata


def build_data_dictionary() -> str:
    lines = [
        "# Data Dictionary\n",
        "Every column after Stage 1 cleaning — raw columns plus everything "
        "`TitanicPreprocessor` adds. Same source as `artifacts/feature_schema.json`.\n",
        "| Feature | Type | Source | Description | Used in Production |",
        "|---|---|---|---|---|",
        "| Survived | Binary | Original dataset | Target label: 1 = survived, 0 = did not survive. | Target (label) |",
    ]
    status_label = {"production": "Yes", "candidate": "Candidate", "deprecated": "Deprecated"}
    for f in FEATURE_CATALOG:
        used = status_label[f["production_status"]]
        lines.append(f"| {f['name']} | {f['type_label']} | {f['source_detail']} | {f['description']} | {used} |")
    lines.append("")
    lines.append(
        "\"Deprecated\" = in the raw dataset but not in the model schema "
        "(PassengerId, Name, Ticket, Cabin) — decomposed into an engineered "
        "feature or dropped as a bare identifier."
    )
    return "\n".join(lines)


def build_pipeline_order_section() -> str:
    return """## Preprocessing Pipeline

**Fit** (train split only, once):
1. Validate schema
2. Extract Title
3. Compute rare-title list (<10 occurrences)
4. Normalize titles (`Rare` collapse, Mlle/Ms/Mme map)
5. Compute per-Title Age median + overall fallback
6. Compute Embarked mode
7. Compute known Deck letters

**Transform** (train, test, and every inference row — reuses fitted values, never recomputes):
1. Validate schema
2. Normalize Title (fitted rare-title list)
3. Impute Age (fitted per-Title medians)
4. Impute Embarked (fitted mode)
5. HasCabin from Cabin
6. FamilySize from SibSp + Parch
7. IsAlone from FamilySize
8. FarePerPerson (guarded against divide-by-zero)
9. Deck from Cabin (fitted known-letter set)
10. AgeGroup by binning Age
11. Reorder columns

**Hard dependencies:** Title → Age → AgeGroup, and FamilySize → IsAlone / FarePerPerson. Everything else is unordered but fixed for determinism.

**Takeaway**

Fit once, transform many times, same fitted values every time.
"""


def build_training_vs_inference_section() -> str:
    return """## Training vs. Inference

**Training** (run once, offline — `python analysis/stage1_data_audit.py` today; `model/train.py` from Stage 4 onward):

```
Raw Data (data/train.csv)
    |
    v
train_test_split (stratified, random_state=42)
    |
    v
Fit TitanicPreprocessor   <-- .fit() runs here, and ONLY here
    |
    v
Transform train/test splits
    |
    v
Train Model
    |
    v
Save artifacts/preprocessing.pkl + preprocessing_metadata.json
    |
    v
Save model/model.pkl
```

**Inference** (run per-request, online — `POST /predict` from Stage 6 onward):

```
Incoming request (single passenger row)
    |
    v
Load artifacts/preprocessing.pkl   <-- once, at API startup, never per-request
    |
    v
preprocessor.transform(request_row)   <-- .transform() ONLY, never .fit() or .fit_transform()
    |
    v
Load model/model.pkl   <-- once, at API startup, never per-request
    |
    v
model.predict(transformed_row)
    |
    v
Return PredictionResponse
```

**Takeaway**

`.fit()` appears exactly once, in Training. If it ever runs inside a request handler, that's a bug.
"""


def build_known_limitations_section() -> str:
    return """## Known Limitations

- **AgeGroup bins are heuristic** — a standard convention, not derived from this dataset's survival curve.
- **Deck depends on Cabin availability** — 77% missing, so its real information content is thin.
- **Statistical significance ≠ predictive usefulness** — a low p-value says "not noise," not "improves the model."
- **Final feature decisions depend on Stage 2 and Stage 3** — every keep-with-caveat verdict here is provisional.
- **`model.pkl` still has the preprocessing leak until retrained** — the fix exists in `artifacts/preprocessing.pkl`, not yet in production.
- **`preprocessing_version` isn't enforced anywhere yet** — recorded now, checked starting Stage 6.
"""


def build_stage1_summary(audit_df: pd.DataFrame, metadata: dict) -> str:
    kept = audit_df[audit_df["decision"] == "keep"]["feature"].tolist()
    caveat = audit_df[audit_df["decision"] == "keep-with-caveat"]["feature"].tolist()
    dropped = audit_df[audit_df["decision"] == "drop"]["feature"].tolist()

    lines = [
        "# Stage 1 Summary — Data Understanding & Cleaning\n",
        "*Full detail in `reports/stage1_data_audit.md`. Feature reference in `reports/data_dictionary.md`.*\n",
        "## What This Stage Did",
        "- Audited `data/train.csv` (891 rows, 12 columns) — shape, dtypes, missingness, duplicates (none found)",
        "- Fit a leakage-safe `TitanicPreprocessor` on an 80/20 stratified split (`random_state=42`)",
        "- Ran a chi-square / point-biserial audit of every feature against `Survived`",
        "- Serialized the fitted preprocessor, a feature contract, and provenance metadata as artifacts\n",
        "## Missing-Value Strategy",
        "- **Age** (19.9% missing) — per-Title median, fit on train split only",
        "- **Embarked** (0.2% missing) — training-split mode",
        "- **Cabin** (77.1% missing) — not imputed, converted to `HasCabin` instead\n",
        "## Engineered Features",
        "`HasCabin`, `FamilySize`, `IsAlone`, `Title` (production) plus three candidates: "
        "`FarePerPerson`, `Deck`, `AgeGroup`.\n",
        f"## Feature Decisions ({len(audit_df)} audited)",
        f"- **Keep ({len(kept)}):** {', '.join(kept) if kept else 'none'}",
        f"- **Keep-with-caveat ({len(caveat)}):** {', '.join(caveat) if caveat else 'none'}",
        f"- **Drop ({len(dropped)}):** {', '.join(dropped) if dropped else 'none'}",
        "",
        "Two caveats worth surfacing here:",
        "- **Deck vs. HasCabin** — near-duplicates. Final call deferred to Stage 3.",
        "- **FamilySize / SibSp** — no linear correlation, but a strong non-monotonic effect. Kept.\n",
        "## Leakage Prevention",
        "Split happens before any fitting. `TitanicPreprocessor.fit()` runs on the train split only "
        "and stores every value it needs; `transform()` only ever reads those stored values back.",
        "",
        "**Takeaway**",
        "",
        "Safe to call on a single inference row — nothing gets recomputed from live traffic. "
        "Full detail in `reports/stage1_data_audit.md`.\n",
        build_pipeline_order_section(),
        build_training_vs_inference_section(),
        "## Robustness Safeguards",
        "Each one addresses a specific way a malformed row could fail deep inside `transform()`:",
        "- Unknown titles → `Rare`",
        "- Unknown deck letters → `Other` (distinct from `U` = no cabin)",
        "- Missing Cabin → `HasCabin=0`, `Deck='U'`",
        "- `FarePerPerson` divide-by-zero guarded",
        "- No chained assignment — copy once, plain column writes",
        "- Column order preserved after `transform()`",
        "- `validate_schema()` fails fast with one clear error\n",
        "**Takeaway**",
        "",
        "Every safeguard maps to a concrete failure mode, not a hypothetical one.\n",
        "## Artifact Versioning",
        f"`preprocessing_metadata.json` records `preprocessing_version` (`{metadata['preprocessing_version']}`), "
        f"`schema_version` (`{metadata['schema_version']}`), `model_version` (`null` — no model versioning "
        f"scheme yet), `creation_timestamp`, `sklearn_version` (`{metadata['sklearn_version']}`), "
        f"`python_version` (`{metadata['python_version']}`), and `git_commit` "
        f"(`{metadata['git_commit'] or 'unavailable'}`).",
        "",
        "**Takeaway**",
        "",
        "Any production issue traces back to an exact commit and fitted instance — no guessing.\n",
        "## Artifacts Generated",
        "- `artifacts/preprocessing.pkl` — fitted `TitanicPreprocessor`",
        f"- `artifacts/feature_schema.json` — full feature contract (schema_version `{FEATURE_SCHEMA_VERSION}`)",
        f"- `artifacts/preprocessing_metadata.json` — fitted values + provenance (created `{metadata['creation_timestamp']}`)",
        "- `reports/stage1_data_audit.md`, `reports/data_dictionary.md`, `reports/stage1_summary.md`\n",
        "## Assumptions",
        "- Fare, SibSp, Parch have zero missing values in `data/train.csv`",
        "- Unseen titles collapse to `Rare` at inference (not left as an unseen category)",
        "- `AgeGroup` bin edges (0/12/18/59/120) are a standard convention, not data-derived",
        "- `git_commit` degrades to `null`, not an error, when git isn't available",
        "- `validate_schema()` treats `PassengerId`, `Survived`, `Ticket` as known-optional, not unexpected\n",
        "## Open Decisions",
        "- `model/train.py` fits Age-by-Title medians on the full dataset before splitting — needs a Stage 4 retrain to fix",
        "- `Deck` vs. `HasCabin`, `IsAlone` vs. `FamilySize` — deferred to Stage 3",
        "- Whether `FarePerPerson` replaces or supplements `Fare` — deferred to Stage 2 VIF",
        "- `statsmodels` added to `requirements.txt` for Stage 2's VIF audit",
        "- `model_version` stays null until Stage 4/6 introduces model versioning\n",
        build_known_limitations_section(),
    ]
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raw = load_raw()
    shape_report = report_shape_and_missingness(raw)

    # Split BEFORE fitting any preprocessing, so nothing about the test rows
    # leaks into the values used to fill missing training rows.
    train_raw, test_raw = train_test_split(
        raw, test_size=0.2, random_state=RANDOM_STATE, stratify=raw["Survived"]
    )

    preprocessor = TitanicPreprocessor().fit(train_raw)
    metadata = write_artifacts(preprocessor, n_train_rows=len(train_raw))

    # For the audit we want every row labelled, so apply the train-fitted
    # preprocessor across the full dataset (same as inference would).
    full = preprocessor.transform(raw)

    audit_df = build_audit_table(full)

    lines = [
        "# Stage 1 — Data Understanding & Cleaning\n",
        shape_report, "",
        "## Missing Value Handling\n",
        "- **Age** (19.9% missing) — median Age per Title, fit on train split only. Title beats a "
        "flat median (Master ≈ 3, Mr ≈ 30).",
        f"- **Embarked** (0.2% missing) — training-split mode (`{preprocessor.embarked_mode_}`).",
        "- **Cabin** (77.1% missing) — not imputed. Converted to `HasCabin`: whether a cabin was "
        "recorded at all carries signal on its own.\n",
        build_leakage_section(preprocessor.embarked_mode_),
        "## Feature Engineering\n",
        "- `FamilySize = SibSp + Parch + 1`, `IsAlone = (FamilySize == 1)` — production.",
        "- `Title` from `Name`; titles <10 occurrences (train split) collapse to `Rare` — production.",
        "- `FarePerPerson = Fare / FamilySize` (candidate) — per-individual price from a group fare.",
        "- `Deck = Cabin[0]`, missing → `'U'` (candidate) — redundant with `HasCabin`, see below.",
        f"- `AgeGroup` (candidate) — `Age` binned into {' / '.join(AGE_GROUP_LABELS)}.\n",
        render_audit_table_markdown(audit_df), "",
        render_verdicts_markdown(audit_df), "",
        build_deck_hascabin_section(),
        build_familysize_section(),
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    DATA_DICTIONARY_PATH.write_text(build_data_dictionary(), encoding="utf-8")
    SUMMARY_PATH.write_text(build_stage1_summary(audit_df, metadata), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n\n[Audit report written to {AUDIT_REPORT_PATH}]")
    print(f"[Data dictionary written to {DATA_DICTIONARY_PATH}]")
    print(f"[Executive summary written to {SUMMARY_PATH}]")
    print(f"[Preprocessing artifact written to {PREPROCESSING_PKL_PATH}]")
    print(f"[Feature schema written to {FEATURE_SCHEMA_PATH}]")
    print(f"[Preprocessing metadata written to {PREPROCESSING_METADATA_PATH}]")


if __name__ == "__main__":
    main()
