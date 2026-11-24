from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "champion_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Credit Default Risk Prediction API",
    description=(
        "A learning-focused REST API serving the trained Logistic Regression "
        "champion pipeline for the synthetic credit-default-risk dataset."
    ),
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    annual_income: float
    loan_amount: float
    credit_score: float
    employment_years: float
    debt_to_income_ratio: float
    loan_term_months: float
    education: Literal["High School", "Graduate", "Postgraduate"]
    employment_type: Literal["Contract", "Salaried", "Self-employed"]
    home_ownership: Literal["Mortgage", "Own", "Rent"]
    loan_purpose: Literal[
        "Education", "Home Improvement", "Medical", "Personal", "Vehicle"
    ]

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probabilities: dict[str, float]

@app.get("/")
def root():
    return {
        "message": "Credit Default Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        input_data = pd.DataFrame([payload.model_dump()])
        prediction = int(model.predict(input_data)[0])
        probabilities_array = model.predict_proba(input_data)[0]
        probabilities = {
            str(int(class_label)): float(probability)
            for class_label, probability in zip(model.classes_, probabilities_array)
        }
        return PredictionResponse(
            prediction=prediction,
            prediction_label="Default" if prediction == 1 else "No Default",
            probabilities=probabilities,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
