import joblib
import pandas as pd

# Load model
model = joblib.load("models/area_risk_model.pkl")

# New area data
new_data = pd.DataFrame([{
    "incident_count": 4,
    "lighting_score": 0.3,
    "crowd_density": 0.4,
    "hour": 22,
    "user_reports_count": 3
}])

# Predict safety score
risk_score = model.predict(new_data)[0]

print("Predicted Safety Score:", round(risk_score, 2))
