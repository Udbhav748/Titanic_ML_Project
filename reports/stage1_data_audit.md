# Stage 1 — Data Understanding & Cleaning

## Data Overview

- Shape: 891 rows x 12 columns
- Duplicate rows: 0
- Duplicate `PassengerId` values: 0

| Column | dtype | missing_count | missing_pct |
|---|---|---|---|
| PassengerId | int64 | 0 | 0.0% |
| Survived | int64 | 0 | 0.0% |
| Pclass | int64 | 0 | 0.0% |
| Name | object | 0 | 0.0% |
| Sex | object | 0 | 0.0% |
| Age | float64 | 177 | 19.87% |
| SibSp | int64 | 0 | 0.0% |
| Parch | int64 | 0 | 0.0% |
| Ticket | object | 0 | 0.0% |
| Fare | float64 | 0 | 0.0% |
| Cabin | object | 687 | 77.1% |
| Embarked | object | 2 | 0.22% |

## Missing Value Handling

- **Age** (19.9% missing) — median Age per Title, fit on train split only. Title beats a flat median (Master ≈ 3, Mr ≈ 30).
- **Embarked** (0.2% missing) — training-split mode (`S`).
- **Cabin** (77.1% missing) — not imputed. Converted to `HasCabin`: whether a cabin was recorded at all carries signal on its own.

## Leakage Prevention

Split happens before anything is fit:

1. `train_test_split` (80/20, stratified, `random_state=42`)
2. `TitanicPreprocessor.fit(train_raw)` — medians, mode, rare-title list computed from the training split only
3. `transform()` reads those fitted values back, never recomputes stats from its input
4. Fitted preprocessor serialized to `artifacts/preprocessing.pkl`
5. Inference loads it once at startup, calls `.transform()` only — never `.fit()`

Recomputing a median at inference time would quietly shift the model's inputs away from what it was trained on. A single request has no group to average over anyway.

**Found while building this:** `model/train.py` fits its Age-by-Title medians on the full dataset, before splitting — a small leak. `TitanicPreprocessor` fixes it (embarked mode fit on train split: `S`); `model/model.pkl` doesn't have the fix until it's retrained.

**Takeaway**

Fit on train only, persist the values, never refit at inference. `model/train.py` needs a retrain to close the gap.

## Feature Engineering

- `FamilySize = SibSp + Parch + 1`, `IsAlone = (FamilySize == 1)` — production.
- `Title` from `Name`; titles <10 occurrences (train split) collapse to `Rare` — production.
- `FarePerPerson = Fare / FamilySize` (candidate) — per-individual price from a group fare.
- `Deck = Cabin[0]`, missing → `'U'` (candidate) — redundant with `HasCabin`, see below.
- `AgeGroup` (candidate) — `Age` binned into Child / Teen / Adult / Senior.

## Statistical Audit

Categorical → chi-square (Cramer's V). Numeric → point-biserial correlation (≡ ANOVA for 2 groups). Sorted by effect size.

| Feature | In prod schema | Test | Statistic | p-value | Effect size | Decision | Caveat |
|---|---|---|---|---|---|---|---|
| Title | yes | chi-square | 284.485 | <0.001 | 0.565 (Cramer's V) | **keep** |  |
| Sex | yes | chi-square | 260.717 | <0.001 | 0.541 (Cramer's V) | **keep** |  |
| Pclass | yes | chi-square | 102.889 | <0.001 | 0.34 (Cramer's V) | **keep** |  |
| Deck | no | chi-square | 99.164 | <0.001 | 0.334 (Cramer's V) | **keep-with-caveat** | 77% of rows fall in the single 'U' (unknown) bucket and several real decks have <5 expected count per cell — chi-square assumptions are shaky. Redundant with HasCabin. |
| HasCabin | yes | chi-square | 87.941 | <0.001 | 0.314 (Cramer's V) | **keep-with-caveat** | Strongly overlaps with Deck (Deck='U' <=> HasCabin=0); keep HasCabin, treat Deck as exploratory only. |
| Fare | yes | point-biserial r (≡ two-group ANOVA/t-test) | 0.257 | <0.001 | 0.257 (|point-biserial r|) | **keep** |  |
| FarePerPerson | no | point-biserial r (≡ two-group ANOVA/t-test) | 0.222 | <0.001 | 0.222 (|point-biserial r|) | **keep-with-caveat** | Derived from Fare / FamilySize; check VIF against both parents in Stage 2 before finalizing. |
| IsAlone | yes | chi-square | 36.001 | <0.001 | 0.201 (Cramer's V) | **keep-with-caveat** | Deterministic function of FamilySize (IsAlone = FamilySize==1); redundant signal, keep only if it linearly separates better than FamilySize alone. |
| Embarked | yes | chi-square | 25.964 | <0.001 | 0.171 (Cramer's V) | **keep** |  |
| AgeGroup | no | chi-square | 14.525 | 0.0023 | 0.128 (Cramer's V) | **keep-with-caveat** | Derived from Age via fixed bins; binning trades continuous signal for interpretability. Compare against raw Age importance in Stage 2. |
| Parch | yes | point-biserial r (≡ two-group ANOVA/t-test) | 0.082 | 0.0148 | 0.082 (|point-biserial r|) | **keep-with-caveat** | Weak on its own (p=0.015, small effect); contributes mainly through FamilySize. Keep-with-caveat. |
| Age | yes | point-biserial r (≡ two-group ANOVA/t-test) | -0.079 | 0.0184 | 0.079 (|point-biserial r|) | **keep** |  |
| SibSp | yes | point-biserial r (≡ two-group ANOVA/t-test) | -0.035 | 0.2922 | 0.035 (|point-biserial r|) | **keep-with-caveat** | Same story as FamilySize: not significant as a linear predictor (p=0.29) but is significant as a categorical one (p<0.001) — non-monotonic, mostly redundant with FamilySize. Keep-with-caveat. |
| FamilySize | yes | point-biserial r (≡ two-group ANOVA/t-test) | 0.017 | 0.6199 | 0.017 (|point-biserial r|) | **keep-with-caveat** | Point-biserial correlation is essentially zero (p=0.62) because the true relationship is non-monotonic — mid-size families (2-4) survived best, solo travelers and large families worst — which a linear test can't see. A chi-square treating it as categorical IS significant (p<0.001). Keep-with-caveat: real signal, wrong linear test. |

## Per-Feature Verdicts

- **Title** — Encodes sex + age + status in one feature. Strongest categorical effect. Expect VIF overlap with Sex/Age in Stage 2.
- **Sex** — Largest effect in the dataset. Women survived ~74% vs. ~19% for men.
- **Pclass** — Strong effect — proxy for socioeconomic status and lifeboat access.
- **Deck** — Significant mostly because 'U' dominates. See Deck vs. HasCabin below.
- **HasCabin** — Solid, low-noise binary signal despite 77% missingness.
- **Fare** — Strong effect, right-skewed. Log-transform candidate for Stage 2.
- **FarePerPerson** — Cleaner separation than raw Fare, but correlated with both parents. VIF decides in Stage 2.
- **IsAlone** — Deterministic threshold on FamilySize — largely redundant. Compare importance in Stage 2.
- **Embarked** — Significant, but likely a Pclass proxy (README Phase 1). Recheck conditional on Pclass in Stage 2.
- **AgeGroup** — Effect driven almost entirely by the Child bucket. Raw Age likely carries more information.
- **Parch** — Weak on its own (p=0.015, small effect). Contributes mainly through FamilySize.
- **Age** — Modest linear effect. Young children survive notably better.
- **SibSp** — Same story as FamilySize — not linear, but significant as categorical. See FamilySize section below.
- **FamilySize** — Linear correlation is ~0 because the relationship is non-monotonic. See section below.

**Takeaway**

Sex, Title, and Pclass dominate. Everything else is kept with a caveat pending Stage 2/3 evidence — nothing is dropped on Stage 1 alone.

## Deck vs. HasCabin

`Deck` is just `Cabin[0]`, with missing Cabin mapped to `'U'`. So `HasCabin=0` and `Deck='U'` carry the same information, and several deck letters have fewer than 10 rows each.

Keeping both risks:
- redundant one-hot columns, ~90% collinear with `HasCabin`
- unstable coefficients/splits from near-duplicate features

Which one survives depends on VIF and feature importance (Stage 2), not a guess made here.

**Takeaway**

Deck and HasCabin overlap almost completely. Final call deferred to Stage 3.

## FamilySize: Linear Test Missed It

Point-biserial correlation is ~0 (r=0.017, p=0.62) — looks like no effect.

But survival by family size isn't linear: solo travelers do worse, families of 2-4 do best, large families do worst. A linear test averages the rise and the fall to zero.

A chi-square treating FamilySize as categorical picks it up clearly (p<0.001).

**Takeaway**

FamilySize carries real signal — the linear test was the wrong tool. Tree models will capture this shape natively; logistic regression won't unless it's binned.
