import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load new customers data (simulate daily data)
df = pd.read_csv("data/new_customers.csv")

# Scale
X = scaler.transform(df)

# Predict churn probability
df['churn_prediction'] = model.predict(X)

# Save output
df.to_csv("data/churn_results.csv", index=False)

print("Batch prediction complete. Check churn_results.csv")
