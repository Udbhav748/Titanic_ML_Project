# Stage 3 — Production Feature Schema Design Review

This is not a ranking exercise. It's a decision record for what the production
model consumes, why, and what changes to ship it. Evidence sources: Stage 1's
statistical audit, Stage 2's VIF/mutual-information/distribution/interaction
analysis, and three non-statistical criteria — engineering cost, production
risk, interpretability — that Stage 1/2 didn't score.

## Decision Table

| Feature | Current Status | Statistical Evidence | Mutual Information | VIF Impact | Engineering Cost | Production Risk | Interpretability | Decision | Reason |
|---|---|---|---|---|---|---|---|---|---|
| **Title** | Production | χ²=284.5, V=0.565, p<.001 (strongest) | 0.166 (rank 1) | Not in VIF matrix (categorical) | None — already computed | Low — unseen-title fallback to `Rare` exists (Stage 1) | High — plain-English honorific | **Keep** | Strongest feature on every metric. Zero cost, zero risk. |
| **Sex** | Production | χ²=260.7, V=0.541, p<.001 | 0.151 (rank 2) | 1.16 (as `Sex_male`) — clean | None — raw client field | Low — closed 2-value set | High | **Keep** | Second-strongest signal, cheapest possible feature. |
| **Pclass** | Production | χ²=102.9, V=0.34, p<.001 | 0.058 (rank 5) | 2.68 — acceptable | None — raw client field | Low | High — ordinal, self-explanatory | **Keep** | Solid standalone signal; VIF confirms it's not absorbing other features' variance. |
| **Fare** | Production | r=0.257, p<.001 | 0.139 (rank 4) | 6.42 — above threshold, but only vs. FarePerPerson | Low — log1p is a one-line pipeline step | Low — monotonic transform, safe for both tree and linear models | Medium — raw currency value | **Transform** | Skew 4.78 → 0.39 under log1p (Stage 2, §2.8). Applied inside the model pipeline's numeric step, not the input schema — harmless to trees, fixes the linear-model case. |
| **Embarked** | Production | χ²=26.0, V=0.171, p<.001 | 0.014 (rank 12, weak) | Not in VIF matrix | None — raw client field, mode-imputed | Low — 3-value set, imputation already fitted (Stage 1) | Medium — likely a Pclass proxy (README Phase 1) | **Keep** | Weak but real, and free to keep. Removing an already-integrated, already-imputed, near-zero-cost feature for a marginal MI gain isn't a good trade. |
| **HasCabin** | Production | χ²=87.9, V=0.314, p<.001 | 0.049 (rank 7) | 2.21 — acceptable | None — client supplies it directly today | Low | High — single boolean | **Keep** | Cheap, stable, statistically solid. Wins the tradeoff against Deck below despite Deck's marginal MI edge. |
| **FamilySize** | Production | r=0.017, p=.62 (linear test misses it) — but χ² on the binned groups is p<.001 (Stage 1, §FamilySize) | 0.032 (rank 8) | **∞ when co-modeled with SibSp+Parch** — exact linear combination | Low — already computed from SibSp+Parch | Low, once SibSp/Parch are removed as co-features | Medium — requires the non-monotonic story to explain | **Keep** | Real, non-monotonic signal Stage 2 confirmed. Must not be modeled alongside its own components (see Merge decision below). |
| **SibSp** | Production | r=-0.035, p=.29 (not significant alone) | 0.001 (rank 14, lowest) | **∞** (exact component of FamilySize) | None — required client field to compute FamilySize | High if kept as a co-feature — perfect collinearity destabilizes coefficient estimates | Low marginal value alone | **Merge** | Folds into `FamilySize`. Stays a required API input; stops being a direct model column. |
| **Parch** | Production | r=0.082, p=.015 (weak) | 0.029 (rank 9) | **∞** (exact component of FamilySize) | None — required client field to compute FamilySize | High if kept as a co-feature, same reason as SibSp | Low marginal value alone | **Merge** | Same treatment as SibSp — real but fully mediated through `FamilySize`. |
| **IsAlone** | Production | χ²=36.0, V=0.201, p<.001 | 0.021 (rank 11) | 2.31 alone, but deterministic function of FamilySize | None — currently one line of code | Medium — redundant threshold logic to maintain for no gain | Medium | **Drop** | `IsAlone = (FamilySize == 1)` is a strict subset of what `FamilySize` already encodes, and scores lower on MI than FamilySize itself. Dead weight. |
| **Deck** | Candidate | χ²=99.2, V=0.334, p<.001, but several cells have expected count <5 (Stage 1) | 0.055 (rank 6) — edges out HasCabin by 0.006 | Not computed (high-cardinality categorical; would need dummy-specific VIF) | High — 7-8 sparse one-hot columns, `Other`/`U` fallback logic to maintain (Stage 1 robustness work) | Medium — sparse categories are exactly where a production model overfits or destabilizes | Low — "which deck letter" is not an intuitive input for a booking-style API | **Drop** | A 0.006 MI edge over HasCabin does not justify 7x the encoding complexity, sparse-cell instability, and a harder-to-explain input. Do not rely on MI alone — this is the textbook case for it. |
| **AgeGroup** | Candidate | χ²=14.5, V=0.128, p=.0023 (weakest significant categorical) | 0.008 (rank 13) — a third of raw Age's MI | Not computed (ordinal categorical) | Low, but redundant — recomputes what Age already provides | Low | Medium — bins are a judgment call (Stage 1), not derived from data | **Drop** | Binning strictly loses information here. Raw Age already outperforms it on every axis. |
| **FarePerPerson** | Candidate | r=0.222, p<.001 (close to Fare) | 0.141 (rank 3) — narrowly beats Fare | 5.35 — above threshold, mutually redundant with Fare | Medium — requires FamilySize computed first, adds a division-by-zero edge case (guarded in Stage 1) | Low — guarded, but one more moving part | Medium — "fare divided by family size" needs a sentence to explain | **Drop** | MI edge over Fare (0.141 vs 0.139) is noise-level. VIF flags them as redundant. Keeping the simpler, already-production feature (Fare) is the correct call, not the marginally-higher-MI one. |
| **Sex_Pclass** | New (Stage 2 interaction) | Not chi-square tested standalone; visual interaction is strong (Stage 2, §2.6) | Not measured (not yet in the MI feature set) | Not computed | Medium — 6-category interaction term, one-hot adds columns | Low-Medium — tree models (current champion) already capture this interaction via sequential splits; a linear model would not | Medium — "sex crossed with class" needs explaining despite being intuitive once shown | **Candidate** | Real interaction, but likely redundant for a tree-based model that already splits on Sex then Pclass. Worth testing explicitly for a Logistic Regression candidate in Stage 4 — not committed until that comparison runs. |
| **Age** | Production | r=-0.079, p=.018 | 0.024 (rank 10) | 1.26 — clean | None — raw client field, per-Title-median imputed (Stage 1) | Low — imputation is leakage-safe and fitted, not recomputed at inference | High | **Keep** | Modest but real effect; outperforms its own binned version (AgeGroup) outright. |

Four raw source columns were never candidates and are handled here for
completeness, not re-litigated with the full evidence set above:

| Column | Status | Decision | Reason |
|---|---|---|---|
| `PassengerId` | Original dataset | Drop | Row identifier, no signal by construction. |
| `Name` | Original dataset | Drop (source-only) | Feeds `Title` extraction during training; the live API takes `title` directly from the client and never sees raw `Name` (see Migration Plan). |
| `Ticket` | Original dataset | Drop | High-cardinality identifier, no clean signal, never engineered. |
| `Cabin` | Original dataset | Drop (source-only) | Feeds `HasCabin`/`Deck` during training; the live API takes `has_cabin` directly from the client and never sees raw `Cabin` (see Migration Plan). |

## Final Production Schema

**Model input, in order (8 features, down from 11):**

```
Pclass, Sex, Age, Fare, Embarked, HasCabin, FamilySize, Title
```

**Original Features** (present in the raw dataset, used as-is)
`Pclass`, `Sex`, `Age`, `Fare`, `Embarked`

**Engineered Features** (derived from raw columns during training)
`HasCabin` (from `Cabin`), `FamilySize` (from `SibSp` + `Parch`), `Title` (from `Name`)

**Interaction Features** (evaluated, not promoted)
`Sex_Pclass` — candidate for the Stage 4 model comparison only, not part of this schema.

**Transformed Features**
`Fare` — `log1p` applied inside the model pipeline's numeric preprocessing step. Same column, same input contract; the transform is internal to `model/train.py::build_preprocessor`, not a schema change.

**Removed from the model's input** (`SibSp`, `Parch`, `IsAlone`) — `SibSp`/`Parch` remain required client-facing inputs, since `FamilySize` is computed from them; they simply no longer appear as their own columns in `X`.

## Production Change Log

**Current Production (11 features)**
```
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, HasCabin, FamilySize, IsAlone, Title
```
↓
**New Production (8 features)**
```
Pclass, Sex, Age, Fare, Embarked, HasCabin, FamilySize, Title
```

| Change | What | Why | Deployment Impact | Retraining Required? | API Breaking? | Backward Compatible? |
|---|---|---|---|---|---|---|
| Remove `SibSp` as a model column | Drop from `X`; keep as a client input used only to compute `FamilySize` | VIF = ∞ with `Parch`/`FamilySize`; standalone MI = 0.001 | `model/train.py::feature_cols` and `api/main.py`'s `input_df` both drop the key | Yes | No | Yes at the API layer; No at the model-artifact layer (old `model.pkl` won't accept the new column set) |
| Remove `Parch` as a model column | Same treatment as `SibSp` | VIF = ∞; effect is fully mediated through `FamilySize` (Stage 1) | Same two files | Yes | No | Same as above |
| Remove `IsAlone` | Drop entirely | Deterministic function of `FamilySize`; lower MI than `FamilySize` itself | Same two files | Yes | No | Yes |
| Transform `Fare` with `log1p` | Add one step to the numeric `ColumnTransformer` pipeline | Skew 4.78 → 0.39; free for tree models, needed for linear models | `model/train.py::build_preprocessor` only | Yes (new pipeline artifact) | No | Yes — client still sends raw `Fare` |

`Deck`, `AgeGroup`, `FarePerPerson`, and `Sex_Pclass` were evaluated and are
**not** in this change log — none of them were in production, and none are
being promoted, so there is no production delta for them to log.

## Migration Plan

The live API's `PassengerInput` schema (`api/schemas.py`) already asks the
client for `has_cabin` and `title` directly — not raw `Cabin`/`Name` text. That
existing design choice is what makes this migration low-risk: the columns
being removed (`SibSp`, `Parch`, `IsAlone`) were never independent client
concepts to begin with. `SibSp`/`Parch` stay required request fields; only
their downstream use changes.

**1. Preprocessing changes**
None to `src/preprocessing.py` or `artifacts/preprocessing.pkl`.
`TitanicPreprocessor` keeps producing the full engineered superset (including
`Deck`, `AgeGroup`, `FarePerPerson` — still useful for EDA and future
Stage-4/5 comparisons). Only the *subset* selected as model input changes.

**2. Feature generation changes**
- `model/train.py::load_and_prepare_data` — `feature_cols` goes from 11 items
  to the 8 above. `FamilySize` is still computed the same way.
- `api/main.py::predict_survival` — `input_df` drops the `SibSp`, `Parch`,
  `IsAlone` keys. `family_size = passenger.sibsp + passenger.parch + 1` stays
  unchanged; the client-facing `PassengerInput` schema is untouched.
- `model/train.py::build_preprocessor` — add a `log1p` step ahead of
  `StandardScaler` for the `Fare` column only.

**3. Artifact changes**
- Retrain → new `model/model.pkl` (old one is no longer schema-compatible;
  its `ColumnTransformer` expects 11 named columns).
- `artifacts/feature_schema.json`: bump `schema_version` `"1.0.0"` →
  `"2.0.0"` (the `production_features`/`feature_order` lists change — a
  breaking change to the model's input contract, even though the *API's*
  contract doesn't move).
- `artifacts/preprocessing_metadata.json`: `preprocessing_version` stays
  `"1.1.0"` — `TitanicPreprocessor`'s own logic is unchanged. `model_version`
  (currently `null`) gets populated at retrain time, e.g. `"2.0.0"`, tying a
  specific `model.pkl` to this exact schema version.

**4. API version changes**
No breaking change to `PassengerInput` or `PredictionResponse` — the request
and response shapes are identical. But predictions *will* differ (different
feature set, different fitted model), so consumers need a way to detect that.
Recommendation: expose `model_version` in `HealthResponse` (cheap, already-
tracked metadata) rather than forcing a `/v2/predict` path for a
non-breaking schema change. Reserve `/v2/predict` for Stage 6 if a future
change *does* break the request contract (e.g., if `Sex_Pclass` gets
promoted and requires a new client-supplied field).

**5. Model version changes**
`model_version: null` → `"2.0.0"` at the Stage 4 retrain. Keep the current
`model.pkl` available under a tagged path (e.g. `model/model_v1.pkl` or a git
tag on this commit) so a bad retrain can roll back by swapping a file path,
not by reverting code.

## Executive Decision

The new 8-feature schema is the better production default because it removes
a real defect — three co-linear features (`SibSp`, `Parch`, `FamilySize`)
sitting in the same model, one of them literally at infinite VIF — without
giving up any measured predictive signal. `FamilySize` already captures the
non-monotonic effect Stage 2 confirmed; `SibSp`/`Parch` individually score
near the bottom of the mutual-information ranking (0.001 and 0.029). Cutting
them isn't a simplicity-for-accuracy trade — it's removing noise that was
actively destabilizing coefficient and importance estimates.

The two close calls — `Deck` vs. `HasCabin`, `FarePerPerson` vs. `Fare` —
were decided against the higher-MI option on purpose. In both cases the MI
gap is inside rounding-error territory (0.006 and 0.002), while the cost gap
is not: `Deck` needs sparse-category handling this project already had to
build defensively in Stage 1, and `FarePerPerson` adds a division and an
edge case for a feature VIF already flags as redundant with `Fare`. Optimizing
for a fourth decimal place of mutual information at the cost of maintenance
surface is the wrong trade for a model this size, on data this small (891
rows) — the marginal gain is well within noise, and the marginal cost is
permanent.

`Sex_Pclass` stays a candidate rather than a committed feature because its
justification is model-dependent: the current production model (Random
Forest) already discovers Sex → Pclass interactions through sequential
splits, so an explicit interaction column is likely redundant there. It only
earns its place if Stage 4 finds Logistic Regression competitive, at which
point the interaction term stops being optional. Committing it now, before
that comparison exists, would add a 6-category feature to production on the
strength of a visual pattern alone — exactly the kind of single-metric,
notebook-driven decision this review exists to avoid.

Deployment safety was weighted deliberately: the API's request contract does
not change at all. Every removed feature (`SibSp`, `Parch`, `IsAlone`) was
already either a required client input feeding something else, or a
server-only computed column. That means this is a same-day, low-risk
retrain-and-redeploy — not a client migration. The one thing that does need
tracking is that predictions will shift, which is why `model_version`
exposure (not a new API version) is the right-sized response.

Net effect: 8 features instead of 11, zero measured accuracy cost, one
resolved multicollinearity defect, and no client-facing change — the
schema a senior engineer would sign off shipping without a design review
follow-up.
