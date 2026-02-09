from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/area_risk_model.pkl")

class AreaInput(BaseModel):
    incident_count: int
    lighting_score: float
    crowd_density: float
    hour: int
    user_reports_count: int

@app.post("/predict")
def predict_risk(data: AreaInput):
    df = pd.DataFrame([data.dict()])
    score = model.predict(df)[0]

    return {
        "safety_score": round(score, 2),
        "risk_level": (
            "High" if score > 70 else
            "Medium" if score > 40 else
            "Low"
        )
    }
