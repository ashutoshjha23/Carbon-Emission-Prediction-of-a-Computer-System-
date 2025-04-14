import os
import pandas as pd
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DT_MODEL_FILE = os.path.join(MODEL_DIR, "decision_tree_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")

# Load data
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()
df = df.drop(columns=[col for col in ["battery percentage", "power plugged", "system uptime (hours)"] if col in df.columns], errors="ignore")

# Features and target
X = df.drop(columns=["total estimated emissions"])
y = df["total estimated emissions"]

# Clean and preprocess
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, thresh=int(0.5 * len(X)))
X.fillna(X.median(), inplace=True)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(scaler, SCALER_FILE)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Save the model
joblib.dump(dt_model, DT_MODEL_FILE)
print(f" Decision Tree model trained and saved to {DT_MODEL_FILE}")

# Evaluate
y_pred_dt = dt_model.predict(X_test)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Regressor Performance:")
print(f"  MAE : {mae_dt:.2f} kg CO2")
print(f"  RMSE: {rmse_dt:.2f} kg CO2")
print(f"  R²  : {r2_dt:.4f}")
