import glob
import os
import joblib
from fastapi import FastAPI
import numpy as np

app = FastAPI()

def get_latest_file(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getctime)

model_path = get_latest_file("models/churn_model_*.pkl")
scaler_path = get_latest_file("models/scaler_*.pkl")

print(f"Loading model: {model_path}")

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
