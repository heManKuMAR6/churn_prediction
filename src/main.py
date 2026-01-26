import os
import joblib
from fastapi import FastAPI
import numpy as np

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

print(f"Loading demo model from: {model_path}")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(data: dict):
    values = np.array(list(data.values())).reshape(1, -1)
    values = scaler.transform(values)
    prediction = model.predict(values)[0]
    return {"churn_prediction": int(prediction)}
