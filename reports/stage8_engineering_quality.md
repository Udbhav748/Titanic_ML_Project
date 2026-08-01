# Stage 8 — Engineering Quality Review

Every number in this document was produced by actually running the suite
below, not estimated. Full test code: `tests/`. CI definition:
`.github/workflows/ci.yml`.

## 1. Testing Strategy

| Layer | What's tested | Why | Where |
|---|---|---|---|
| Preprocessing | `TitanicPreprocessor.fit()`/`.transform()`, imputation values, robustness safeguards | This is the single component every other layer depends on — a silent bug here corrupts training, evaluation, and inference alike | `tests/unit/test_preprocessing.py` |
| Schema validation | `validate_schema()` — missing/unexpected columns, wrong dtypes, non-null violations | It's the fail-fast gate meant to catch malformed input before it reaches `fit()`/`transform()` internals; if it doesn't actually fail fast, it's dead code | `tests/unit/test_validate_schema.py` |
| Feature engineering | `FamilySize`, `Title`, `HasCabin`, `Deck`, `AgeGroup`, `FarePerPerson` correctness | Each is a specific, checkable transformation rule (e.g. `FamilySize = SibSp + Parch + 1`) — worth asserting literally, not just "it runs" | `tests/unit/test_preprocessing.py` |
| Artifacts | `.pkl`/`.json` files load, have required keys, and **agree with each other** | Stage 7 shipped a real version-drift bug between two JSON files; this is the regression suite for that class of bug | `tests/unit/test_artifacts.py` |
| Model | `model_v2.pkl` accepts preprocessor output, returns valid probabilities, gets known easy cases right | Confirms the model and the preprocessing contract actually fit together, not just that each loads | `tests/integration/test_preprocessing_to_model.py` |
| API | `/predict`, `/health`, validation errors, response shape | The only externally-facing contract in this system — has to be tested as a black box, not by calling internal functions | `tests/integration/test_api.py` |
| Dashboard | Data/model loading layer (`dashboard/data_utils.py`) | Dashboard doesn't call the API (see Stage 6) — it loads artifacts directly, so that loading layer is the thing that needs verifying | `tests/integration/test_dashboard_data_utils.py` |
| End-to-end | Raw input → preprocessing → prediction, consistency across call paths | Verifies the parts that are supposed to agree actually do, and documents the one part that's expected not to yet (see §4) | `tests/e2e/test_full_pipeline.py` |

Deliberately **not** unit-tested: `analysis/stage1_data_audit.py` and
`analysis/stage4_model_comparison.py`'s `main()`/report-generation
functions. These are one-off scripts that produce markdown reports, run
once per stage, and already validated by inspecting their actual output in
Stages 1 and 4. Their *reusable* logic (`build_candidate_preprocessor`,
`load_candidate_split`) is exercised indirectly through the integration
tests that import it — that's the part other code actually depends on.

## 2. Unit Tests

**47 tests**, `tests/unit/`. Examples of what "meaningful" means here, not trivial "does it run":

- `test_age_imputed_by_title_not_global_median` — asserts a `Master` with missing age gets the *Master* median, not some flat average. Tests the actual design decision from Stage 1, not just "no exception raised."
- `test_unseen_deck_letter_maps_to_other` / `test_unseen_title_collapses_to_rare` — tests the Stage 1 robustness safeguards against exactly the input they were built for (a category never seen during fit).
- `test_error_message_lists_every_problem_at_once` — `validate_schema()` is supposed to report every issue in one pass, not fail on the first and hide the rest; this asserts that property directly.
- `test_fit_transform_matches_separate_fit_and_transform` — the convenience method must be provably identical to calling the two steps by hand, not just "probably fine."
- `test_transform_does_not_mutate_input` — a caller's DataFrame must come back untouched; silent mutation is a classic source of hard-to-reproduce bugs.

**Coverage: `src/preprocessing.py` — 100%** (99/99 statements). This is the
one module every other layer depends on, so full coverage there was the
bar, not padding for a number.

## 3. Integration Tests

**25 tests**, `tests/integration/`.

- **Preprocessing → Model** (`test_preprocessing_to_model.py`, 6 tests) — feeds real `TitanicPreprocessor` output straight into `model_v2.pkl`, including rows that require imputation and rows with unseen categories. Includes two sanity checks against domain knowledge: a 1st-class woman with a cabin must score >0.7, a 3rd-class alone man must score <0.3.
- **Model → API** (`test_api.py`, 14 tests) — `FastAPI TestClient` against the real `app` object, no mocks. Covers the happy path, every required field missing one at a time, invalid enum values, and that optional fields actually default.
- **API → Dashboard** (`test_dashboard_data_utils.py`, 5 tests) — the dashboard doesn't call the API (Stage 6 architecture note); it loads `artifacts/preprocessing.pkl` and `model/model_v2.pkl` directly. So this suite tests that loading layer, including that the dashboard's live-computed metrics reproduce Stage 4/5's already-published numbers to 3 decimal places.

All use realistic sample data (fixtures in `tests/conftest.py`) — a
1st-class woman with a cabin, a 3rd-class man with missing age/embarked/cabin,
a passenger with a title the fitted model never saw — not empty/placeholder
inputs.

## 4. End-to-End Validation

**4 tests**, `tests/e2e/test_full_pipeline.py`.

```
Raw Input → Preprocessing → Prediction → API Response → Dashboard
```

What's verified:
- **Repeatability** — the same raw passenger through the same pipeline twice gives bit-identical output.
- **Preprocessing → Dashboard consistency** — the dashboard's cached loaders and a fresh, direct `TitanicPreprocessor` + `model_v2.pkl` call agree exactly on the same input.
- **API well-formedness** — `/predict` returns a valid probability for the same passenger.

**What's explicitly *not* asserted, and why:** the API (`/predict`, `model_v1.pkl`,
11 features) and the dashboard (`model_v2.pkl`, 8 features) are different
models today. Stage 6 planned the cutover; it hasn't happened. A test that
asserted their probabilities match would be asserting something false and
either fail correctly (confusing) or get quietly weakened until it passed
(worse). Instead, `test_api_v1_and_pipeline_v2_predictions_are_independently_valid_but_not_required_to_match`
checks each is independently valid and that they at least agree on the
qualitative call for an unambiguous case — documenting the real gap instead
of hiding it behind a green checkmark.

## 5. Versioning Strategy

| What | Where tracked | Current value |
|---|---|---|
| Preprocessing | `preprocessing_version` in `preprocessing_metadata.json` | `1.1.0` |
| Model | filename (`model_v1.pkl`, `model_v2.pkl`) + `model_version` in `preprocessing_metadata.json` | `2.0.0` |
| Feature schema | `schema_version` in **both** `feature_schema.json` and `preprocessing_metadata.json` | `2.0.0` / `2.0.0` |
| API | FastAPI `app` `version=` field | `1.0.0` (unchanged — no breaking contract change yet) |

**Synchronization mechanism:** `schema_version` is intentionally recorded in
two files (the feature contract and the fitted-artifact metadata) rather
than one, so a promotion script that updates only one of them produces a
*detectable* inconsistency instead of a silent one. `tests/unit/test_artifacts.py::test_schema_version_matches_across_artifacts`
asserts they agree; the CI pipeline runs this on every push.

**This is not a hypothetical safeguard.** During Stage 7, `model/train_v2.py`'s
promotion step updated `feature_schema.json`'s `schema_version` to `2.0.0`
but left `preprocessing_metadata.json`'s own copy of the same field at
`1.0.0` — a real, shipped bug, caught by inspection while building the
dashboard's Home page, not by any automated check (none existed yet). That
incident is the direct justification for §6 and §7 below: the fix wasn't
just correcting the two files, it was adding a test and a CI step so the
same class of bug fails a pipeline instead of waiting to be noticed.

## 6. CI/CD Pipeline

`.github/workflows/ci.yml`, runs on push/PR. One job, six steps — no
matrix builds, no deployment automation, nothing beyond what was asked for:

1. Install dependencies (`requirements-dev.txt`)
2. Lint (`ruff check`)
3. Build preprocessing/schema artifacts (`analysis/stage1_data_audit.py`)
4. Build and promote `model_v2.pkl` (`model/train_v2.py`)
5. Validate artifacts exist on disk
6. Verify `schema_version` agrees across artifacts + `model_version` is set
7. Run the test suite (`pytest --cov=src`)

**Artifacts are rebuilt in CI, not restored from committed binaries.**
`model_v1.pkl`, `model_v2.pkl`, and `artifacts/*` are reproducible from
`data/train.csv` plus a fixed `random_state=42` — rebuilding them on every
run is both simpler than managing binary files in git and a stronger
guarantee: if the pipeline can't reproduce its own artifacts from source,
that's a real problem CI should catch, not paper over. This whole pipeline
was run locally end-to-end from a clean-room state (artifacts deleted,
rebuilt, tested) before being written up here — it is not speculative.

## 7. Quality Assurance

**Tests:** 76 / 76 passing (47 unit, 25 integration, 4 e2e).
**Lint:** 0 issues (`ruff check`, `E`/`F`/`I`/`W` rule set; 8 real issues found and fixed — unused imports, unsorted imports, trailing whitespace).
**Coverage:** `src/preprocessing.py` 100% (99/99 statements) — the module every other layer depends on.

**Known limitations:**
- API still serves `model_v1.pkl` (11 features); `model_v2.pkl` is validated but not cut over (Stage 6 plan, not yet executed).
- `analysis/*.py` report-generation code has no direct unit tests — validated by output inspection, not automated assertions.
- No automated dashboard UI tests — Stage 7's Playwright verification was run manually, not checked into CI.
- Test suite runs against a single fixed train/test split; it does not re-verify Stage 4's cross-validation folds.

**Remaining technical debt:**
- `/v2/predict` endpoint doesn't exist yet — required before API cutover.
- No automated rollback drill — Stage 6's rollback runbook has never been executed, only documented.
- Dashboard has no test coverage for its Streamlit UI layer, only its data-loading functions.
- `httpx`/`starlette` emit a deprecation warning under the current pinned versions (harmless today, worth revisiting on the next dependency bump).

## 8. Engineering Checklist

- [x] Unit tests for preprocessing, validation, feature engineering, artifacts
- [x] Integration tests for preprocessing→model, model→API, API→dashboard
- [x] End-to-end pipeline test, with the real API/dashboard version gap documented rather than hidden
- [x] 100% test coverage on the core preprocessing module
- [x] Lint clean (ruff)
- [x] CI pipeline: install → lint → build artifacts → validate → verify version compatibility → test
- [x] CI pipeline verified locally end-to-end from a clean-room state
- [x] Versioning strategy documented, with a real caught bug as evidence it's needed
- [ ] API cut over to `model_v2.pkl` / `/v2/predict`
- [ ] Automated dashboard UI tests
- [ ] Rollback runbook executed at least once, not just documented

**Is this repository ready for production deployment?**

**Yes, for the model and preprocessing layer. Not yet for the full system cutover.**

Evidence for yes: the component every other layer depends on
(`TitanicPreprocessor`) has 100% test coverage, the model's contract with
that preprocessing is integration-tested against realistic and adversarial
inputs, the API is tested as a black box including its failure modes, and
a real versioning bug was caught, fixed, and turned into a permanent
regression test plus a CI gate — that's the loop a production system is
supposed to have.

Evidence for not-yet: `/predict` still serves the old model. Deploying
today means production keeps serving `model_v1.pkl` regardless of how
well-tested `model_v2.pkl` is, because the cutover itself — the one
unchecked item that matters most — hasn't happened. That's not a testing
gap; it's the one remaining task on Stage 6's own checklist, and this
stage's tests exist specifically to make that cutover safe when it happens.
