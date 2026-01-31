
import pandas as pd
import numpy as np
import joblib
import os
os.makedirs("models", exist_ok=True)

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE


# 1. Load raw data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "sample_data", "Churn_Modelling.csv")

df = pd.read_csv(data_path)


# 2. Drop useless columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 3. Encode categorical columns
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# 4. Split features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Handle imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 8. Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_sm, y_train_sm)

# 9. Evaluate
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)

print(f"New Model F1 Score: {f1}")

# 10. Versioned model saving
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/churn_model_{timestamp}.pkl"
scaler_path = f"models/scaler_{timestamp}.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Model saved at {model_path}")
print(f"Scaler saved at {scaler_path}")
