# Stage 6 — Production Deployment & Migration Plan

Audience: an engineer deploying the Stage 4/5 model (Logistic Regression, 8
features) to replace the current production model (Random Forest, 11
features) without breaking `api/main.py`'s existing consumers.

## 1. Architecture Overview

**Current (as deployed today):**

```
Raw Request (JSON)
    |
    v
Pydantic Validation (PassengerInput)
    |
    v
Inline feature construction (api/main.py builds the DataFrame by hand)
    |
    v
Load Model (model/model.pkl — unversioned path, loaded once at import)
    |
    v
model.predict_proba()
    |
    v
Return JSON (PredictionResponse)
```

`artifacts/preprocessing.pkl` is **not in this path today**. `api/main.py`
hand-constructs the 11-column DataFrame (`Pclass, Sex, Age, SibSp, Parch,
Fare, Embarked, HasCabin, FamilySize, IsAlone, Title`) directly from the
request, duplicating logic that already exists in `TitanicPreprocessor`.

**Target (this migration):**

```
Raw Request (JSON)
    |
    v
Pydantic Validation (PassengerInput — unchanged)
    |
    v
Load Preprocessing Artifact (artifacts/preprocessing.pkl, loaded once at startup)
    |
    v
TitanicPreprocessor.transform() — engineered feature row
    |
    v
Select production_features (artifacts/feature_schema.json)
    |
    v
Load Model (model/model_v2.pkl, loaded once at startup)
    |
    v
model.predict_proba()
    |
    v
Build Response (survived, survival_probability, model_version)
    |
    v
Return JSON
```

Two changes worth flagging to whoever reviews this: (1) preprocessing moves
from ad hoc to artifact-driven — closes a duplication gap, not just a model
swap; (2) both artifacts load once at process startup, never per-request —
consistent with Stage 1's leakage rule (`.transform()` only, never `.fit()`).

## 2. Model Versioning

| | Path | Algorithm | Schema | Status |
|---|---|---|---|---|
| Current | `model/model.pkl` | Random Forest | 11 features | Frozen as `model_v1.pkl` before cutover |
| New | `model/model_v2.pkl` | Logistic Regression | 8 features (Stage 3) | Candidate, validated in Stage 4/5 |

**Why versioning is required:**
- Rollback without a code change or rebuild
- An audit trail tying a specific `.pkl` file to specific evaluation results (Stage 4/5)
- Prevents an accidental overwrite of a known-good artifact
- Enables running two models side by side during migration (§4)

**How old models remain available:** `model_v1.pkl` is never deleted. Every
future retrain adds `model_vN.pkl`; nothing is overwritten in place.

**How rollback works:** the active model is selected by an environment
variable, not a hardcoded path:

```python
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v2")
MODEL_PATH = BASE_DIR / "model" / f"model_{MODEL_VERSION}.pkl"
```

Rollback = redeploy the same image with `MODEL_VERSION=v1`. No rebuild if
both artifacts already shipped in the image (§5).

## 3. Preprocessing Versioning

`TitanicPreprocessor`'s logic did not change in this migration — Stage 3
only changed which of its output columns the model consumes. There is one
`artifacts/preprocessing.pkl`, tracked by the `preprocessing_version` field
inside `artifacts/preprocessing_metadata.json` (currently `"1.1.0"`), not by
a second file.

**Model and preprocessing versions must always be paired** — a model
trained on the 8-feature schema fed an 11-column DataFrame (or vice versa)
fails or silently mispredicts. The pairing contract:

| Model | Required `preprocessing_version` | Required `schema_version` |
|---|---|---|
| `model_v1.pkl` | n/a (no artifact — inline logic) | n/a (implicit 11-feature schema) |
| `model_v2.pkl` | `>= 1.1.0` | `2.0.0` |

**Enforcement, not just documentation** — add a startup assertion:

```python
metadata = json.load(open(PREPROCESSING_METADATA_PATH))
schema = json.load(open(FEATURE_SCHEMA_PATH))
assert metadata["model_version"] == EXPECTED_MODEL_VERSION, "preprocessing/model version mismatch"
```

If this fails, the process should refuse to serve traffic (`/health`
reports `"degraded"`) rather than silently predict on a mismatched schema.

## 4. API Versioning

**Do not replace `/predict`.** Introduce `/v2/predict` alongside it.

| | `/predict` | `/v2/predict` |
|---|---|---|
| Model | `model_v1.pkl` (Random Forest) | `model_v2.pkl` (Logistic Regression) |
| Request schema | `PassengerInput` | `PassengerInput` — **unchanged, same model** |
| Response schema | `PredictionResponse` | `PredictionResponse` (adds `model_version` field) |

Request schema is identical on purpose — Stage 3 confirmed no client field
was added or removed. `PredictionResponse` gains `model_version: str`; this
is additive and does not break existing `/predict` consumers who ignore
unknown fields.

**Compatibility:** `/predict` keeps serving `model_v1.pkl` unchanged for
the full migration window — it is not silently repointed at the new model.

**Migration timeline:**

| Phase | Action |
|---|---|
| Week 1–2 | Both endpoints live. `/v2/predict` available for opt-in testing only. |
| Week 3–4 | Internal consumers (dashboard) migrate to `/v2/predict`. Monitor prediction distribution vs. `/predict` for drift. |
| Week 5–6 | Announce `/predict` deprecation date to any external consumers. |
| Week 8 | `/predict` returns a `Deprecation` header; `model_v1.pkl` stays loaded. |
| Week 12 | Remove `/predict` route (or repoint it to `model_v2.pkl` if no external consumers remain) — separate decision, not automatic. |

## Why the API Did Not Change

**Client contract remains stable.** Every field in `PassengerInput` —
`pclass, sex, age, sibsp, parch, fare, embarked, has_cabin, title` — is
unchanged. Clients send the exact same payload to `/v2/predict` as they do
to `/predict`. No client-side code changes, no added or removed fields, no
broken integrations.

**Feature engineering is server-side, not client-side.** The client's job
is to supply the raw inputs a booking form would naturally have; it was
never the client's job to compute `FamilySize`, or to know that `Title` and
`HasCabin` get derived rather than supplied on other systems. Stage 1 built
`TitanicPreprocessor` specifically so this logic lives in one place, inside
the prediction service, rather than duplicated in every caller.

**What actually changed is entirely internal:** the preprocessing artifact
wiring, the feature subset selected (`feature_schema.json`), the trained
model file, and the model version. None of these are visible in the request
or response shape a client depends on.

**This is backward compatible by design, not by coincidence.** Existing
applications keep working with zero code changes because the public
contract — request fields, response fields, status codes — was preserved
even though everything behind it changed. That's the point of versioning
the model and the preprocessing artifact separately from the API: it lets
the internals move independently of the interface. Preserving that contract
prevents a breaking change from propagating to every downstream consumer,
keeps this deployment's blast radius contained to the service itself, and
lets consumers migrate to `/v2/predict` on their own schedule instead of on
this team's.

**Takeaway.** The model changed, the schema changed, the preprocessing path
changed — and no client had to know any of that happened. A production
upgrade that requires every consumer to change their code at the same time
isn't a upgrade, it's a coordinated outage waiting for a schedule slip. This
migration was designed so the interface is the one thing that didn't have
to move.

## 5. Deployment Changes

**Docker rebuild:**
```dockerfile
COPY model/model_v1.pkl ./model/model_v1.pkl
COPY model/model_v2.pkl ./model/model_v2.pkl
COPY artifacts/ ./artifacts/
```
Both model versions and the full `artifacts/` directory ship in every
image — version selection happens at runtime via `MODEL_VERSION`, not at
build time. One image serves either version.

**Artifact replacement (before rebuild):**
```
cp model/model.pkl model/model_v1.pkl      # freeze current production
<retrain per Stage 4> -> model/model_v2.pkl
```
`model/model.pkl` (the bare, unversioned name) is retired from the deploy
path after this — `api/main.py` reads `model_{MODEL_VERSION}.pkl` only.

**Environment variables:**
| Variable | Values | Default | Purpose |
|---|---|---|---|
| `MODEL_VERSION` | `v1`, `v2` | `v2` | Selects which model file loads |
| `PREPROCESSING_PATH` | file path | `artifacts/preprocessing.pkl` | Override for testing |

**Startup sequence:**
1. Read env vars
2. Load `artifacts/preprocessing.pkl`
3. Load `artifacts/preprocessing_metadata.json` + `feature_schema.json`
4. Load `model/model_{MODEL_VERSION}.pkl`
5. Assert version pairing (§3) — abort startup on mismatch, don't degrade silently
6. Mark `/health` ready

**Health check** — extend `HealthResponse` to report what's actually
loaded, not just a boolean:
```json
{"status": "ok", "model_loaded": true, "model_version": "2.0.0",
 "preprocessing_version": "1.1.0", "schema_version": "2.0.0"}
```

**EC2 deployment steps** (single-instance, matches the current README
Phase 7 setup — no load balancer in front today):
1. `git pull` on the instance
2. `docker build -t titanic-api:v2 .`
3. Start the new container on a spare port for a pre-cutover check: `docker run -d -p 8001:8000 -e MODEL_VERSION=v2 --name titanic-api-v2 titanic-api:v2`
4. Hit `http://<host>:8001/health` — confirm `model_version: "2.0.0"` and no version-mismatch failure
5. `docker stop titanic-api-container && docker rm titanic-api-container`
6. `docker run -d -p 8000:8000 -e MODEL_VERSION=v2 --name titanic-api-container titanic-api:v2`
7. Remove the temporary container on 8001

**On downtime:** steps 5–6 have a real gap of a few seconds today — this
instance has no reverse proxy in front of it. That gap is not eliminable
without adding nginx/Caddy or a load balancer in front of two containers
and flipping upstream — worth doing before the next migration, out of scope
for this one. Step 3–4's pre-cutover check on a spare port is what actually
catches a bad artifact before it reaches the real port, which is the
higher-value protection here.

## 6. Rollback Strategy

Run this if `/v2/predict` (or `/predict` after full cutover) misbehaves
post-deploy:

1. `docker stop titanic-api-container && docker rm titanic-api-container`
2. `docker run -d -p 8000:8000 -e MODEL_VERSION=v1 --name titanic-api-container titanic-api:v2`
   (same image — `model_v1.pkl` already shipped in it; no rebuild)
3. `curl http://<host>:8000/health` — confirm `model_version: "1.0.0"`, `status: "ok"`
4. Re-run the Stage 5 test-set predictions against the running endpoint — confirm they match the original Stage 4 baseline numbers (prediction parity, not just a 200 response)
5. Notify whoever owns `/v2/predict` consumers that the rollback happened and why
6. File the root cause before attempting cutover again — don't retry blind

## 7. Production Checklist

- [ ] `model_v1.pkl` frozen from current `model.pkl` before any overwrite
- [ ] `model_v2.pkl` trained and matches Stage 4/5's reported metrics exactly
- [ ] `artifacts/feature_schema.json` — `schema_version` bumped to `2.0.0`
- [ ] `artifacts/preprocessing_metadata.json` — `model_version` set to `2.0.0`
- [ ] Startup version-pairing assertion added and tested against a deliberately mismatched pair
- [ ] `/v2/predict` added; `/predict` untouched and still serving `model_v1.pkl`
- [ ] `PredictionResponse.model_version` field added (additive, non-breaking)
- [ ] `HealthResponse` extended with version fields
- [ ] Dockerfile ships both `model_v1.pkl` and `model_v2.pkl` plus `artifacts/`
- [ ] New image built and tagged (`titanic-api:v2`)
- [ ] Pre-cutover check passed on spare port (§5, step 3–4)
- [ ] `/health` verified on the real port after cutover
- [ ] Prediction parity checked against Stage 4/5 test-set numbers
- [ ] Rollback procedure executed once in staging, confirmed working, before relying on it in production
- [ ] Migration timeline (§4) communicated to any external `/predict` consumers
