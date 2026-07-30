from typing import Literal

from pydantic import BaseModel, Field


class PassengerInput(BaseModel):
    pclass: Literal[1, 2, 3] = Field(..., description="Passenger class: 1 = First, 2 = Second, 3 = Third")
    sex: Literal["male", "female"]
    age: float = Field(..., ge=0, le=100)
    sibsp: int = Field(..., ge=0, description="Siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Parents/children aboard")
    fare: float = Field(..., ge=0)
    embarked: Literal["C", "Q", "S"] = Field(default="S", description="Port of embarkation")
    has_cabin: bool = False
    title: Literal["Mr", "Mrs", "Miss", "Master", "Rare"] = Field(
        default="Mr", description="Title from name, e.g. Mr/Mrs/Miss/Master/Rare"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pclass": 1,
                "sex": "female",
                "age": 29.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 100.0,
                "embarked": "C",
                "has_cabin": True,
                "title": "Mrs",
            }
        }


class PredictionResponse(BaseModel):
    survived: bool
    survival_probability: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
