# Stage 1 Summary — Data Understanding & Cleaning

*Full detail in `reports/stage1_data_audit.md`. Feature reference in `reports/data_dictionary.md`.*

## What This Stage Did
- Audited `data/train.csv` (891 rows, 12 columns) — shape, dtypes, missingness, duplicates (none found)
- Fit a leakage-safe `TitanicPreprocessor` on an 80/20 stratified split (`random_state=42`)
- Ran a chi-square / point-biserial audit of every feature against `Survived`
- Serialized the fitted preprocessor, a feature contract, and provenance metadata as artifacts

## Missing-Value Strategy
- **Age** (19.9% missing) — per-Title median, fit on train split only
- **Embarked** (0.2% missing) — training-split mode
- **Cabin** (77.1% missing) — not imputed, converted to `HasCabin` instead

## Engineered Features
`HasCabin`, `FamilySize`, `IsAlone`, `Title` (production) plus three candidates: `FarePerPerson`, `Deck`, `AgeGroup`.

## Feature Decisions (14 audited)
- **Keep (6):** Title, Sex, Pclass, Fare, Embarked, Age
- **Keep-with-caveat (8):** Deck, HasCabin, FarePerPerson, IsAlone, AgeGroup, Parch, SibSp, FamilySize
- **Drop (0):** none

Two caveats worth surfacing here:
- **Deck vs. HasCabin** — near-duplicates. Final call deferred to Stage 3.
- **FamilySize / SibSp** — no linear correlation, but a strong non-monotonic effect. Kept.

## Leakage Prevention
Split happens before any fitting. `TitanicPreprocessor.fit()` runs on the train split only and stores every value it needs; `transform()` only ever reads those stored values back.

**Takeaway**

Safe to call on a single inference row — nothing gets recomputed from live traffic. Full detail in `reports/stage1_data_audit.md`.

## Preprocessing Pipeline

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

## Training vs. Inference

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

## Robustness Safeguards
Each one addresses a specific way a malformed row could fail deep inside `transform()`:
- Unknown titles → `Rare`
- Unknown deck letters → `Other` (distinct from `U` = no cabin)
- Missing Cabin → `HasCabin=0`, `Deck='U'`
- `FarePerPerson` divide-by-zero guarded
- No chained assignment — copy once, plain column writes
- Column order preserved after `transform()`
- `validate_schema()` fails fast with one clear error

**Takeaway**

Every safeguard maps to a concrete failure mode, not a hypothetical one.

## Artifact Versioning
`preprocessing_metadata.json` records `preprocessing_version` (`1.1.0`), `schema_version` (`1.0.0`), `model_version` (`null` — no model versioning scheme yet), `creation_timestamp`, `sklearn_version` (`1.9.0`), `python_version` (`3.13.14`), and `git_commit` (`a3b4418b0c059f52750b4397d099bec3e59fbc39`).

**Takeaway**

Any production issue traces back to an exact commit and fitted instance — no guessing.

## Artifacts Generated
- `artifacts/preprocessing.pkl` — fitted `TitanicPreprocessor`
- `artifacts/feature_schema.json` — full feature contract (schema_version `1.0.0`)
- `artifacts/preprocessing_metadata.json` — fitted values + provenance (created `2026-07-31T20:28:23.933129+00:00`)
- `reports/stage1_data_audit.md`, `reports/data_dictionary.md`, `reports/stage1_summary.md`

## Assumptions
- Fare, SibSp, Parch have zero missing values in `data/train.csv`
- Unseen titles collapse to `Rare` at inference (not left as an unseen category)
- `AgeGroup` bin edges (0/12/18/59/120) are a standard convention, not data-derived
- `git_commit` degrades to `null`, not an error, when git isn't available
- `validate_schema()` treats `PassengerId`, `Survived`, `Ticket` as known-optional, not unexpected

## Open Decisions
- `model/train.py` fits Age-by-Title medians on the full dataset before splitting — needs a Stage 4 retrain to fix
- `Deck` vs. `HasCabin`, `IsAlone` vs. `FamilySize` — deferred to Stage 3
- Whether `FarePerPerson` replaces or supplements `Fare` — deferred to Stage 2 VIF
- `statsmodels` added to `requirements.txt` for Stage 2's VIF audit
- `model_version` stays null until Stage 4/6 introduces model versioning

## Known Limitations

- **AgeGroup bins are heuristic** — a standard convention, not derived from this dataset's survival curve.
- **Deck depends on Cabin availability** — 77% missing, so its real information content is thin.
- **Statistical significance ≠ predictive usefulness** — a low p-value says "not noise," not "improves the model."
- **Final feature decisions depend on Stage 2 and Stage 3** — every keep-with-caveat verdict here is provisional.
- **`model.pkl` still has the preprocessing leak until retrained** — the fix exists in `artifacts/preprocessing.pkl`, not yet in production.
- **`preprocessing_version` isn't enforced anywhere yet** — recorded now, checked starting Stage 6.
