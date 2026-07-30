from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

from api.schemas import HealthResponse, PassengerInput, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predicts passenger survival probability using a trained model.",
    version="1.0.0",
)

model = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model_load_error = str(exc)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_survival(passenger: PassengerInput) -> PredictionResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available: {model_load_error}",
        )

    family_size = passenger.sibsp + passenger.parch + 1

    input_df = pd.DataFrame([{
        "Pclass": passenger.pclass,
        "Sex": passenger.sex,
        "Age": passenger.age,
        "SibSp": passenger.sibsp,
        "Parch": passenger.parch,
        "Fare": passenger.fare,
        "Embarked": passenger.embarked,
        "HasCabin": int(passenger.has_cabin),
        "FamilySize": family_size,
        "IsAlone": int(family_size == 1),
        "Title": passenger.title,
    }])

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
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."},
    )
