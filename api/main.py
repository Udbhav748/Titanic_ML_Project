import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

from api.schemas import HealthResponse, PassengerInput, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model_v2.pkl"
FEATURE_SCHEMA_PATH = BASE_DIR / "artifacts" / "feature_schema.json"
PREPROCESSING_METADATA_PATH = BASE_DIR / "artifacts" / "preprocessing_metadata.json"

# The pairing this API was built against — see reports/stage6_deployment_plan.md
# section 3. A mismatch here means the artifacts on disk don't match what this
# code expects, so the API refuses to serve rather than silently mispredict.
EXPECTED_SCHEMA_VERSION = "2.0.0"
EXPECTED_MODEL_VERSION = "2.0.0"

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predicts passenger survival probability using a trained model.",
    version="2.0.0",
)

model = None
model_load_error = None
feature_order: list[str] | None = None
preprocessing_version = None
schema_version = None
try:
    model = joblib.load(MODEL_PATH)
    feature_schema = json.load(open(FEATURE_SCHEMA_PATH))
    preprocessing_metadata = json.load(open(PREPROCESSING_METADATA_PATH))
    schema_version = feature_schema["schema_version"]
    preprocessing_version = preprocessing_metadata["preprocessing_version"]
    if (
        preprocessing_metadata["model_version"] != EXPECTED_MODEL_VERSION
        or schema_version != EXPECTED_SCHEMA_VERSION
    ):
        raise ValueError(
            f"preprocessing/model version mismatch: expected model_version="
            f"{EXPECTED_MODEL_VERSION!r} schema_version={EXPECTED_SCHEMA_VERSION!r}, "
            f"got model_version={preprocessing_metadata['model_version']!r} "
            f"schema_version={schema_version!r}"
        )
    feature_order = feature_schema["feature_order"]
except Exception as exc:
    model = None
    model_load_error = str(exc)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=EXPECTED_MODEL_VERSION if model is not None else None,
        preprocessing_version=preprocessing_version,
        schema_version=schema_version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_survival(passenger: PassengerInput) -> PredictionResponse:
    """Logistic Regression, 8-feature schema (artifacts/feature_schema.json) —
    the model selected in Stage 4/5 and shown throughout the dashboard."""
    if model is None or feature_order is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available: {model_load_error}",
        )

    family_size = passenger.sibsp + passenger.parch + 1
    raw = {
        "Pclass": passenger.pclass,
        "Sex": passenger.sex,
        "Age": passenger.age,
        "Fare": passenger.fare,
        "Embarked": passenger.embarked,
        "HasCabin": int(passenger.has_cabin),
        "FamilySize": family_size,
        "Title": passenger.title,
    }
    input_df = pd.DataFrame([{col: raw[col] for col in feature_order}])

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )

    return PredictionResponse(
        survived=bool(prediction),
        survival_probability=round(float(probability), 4),
        model_version=EXPECTED_MODEL_VERSION,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."},
    )
