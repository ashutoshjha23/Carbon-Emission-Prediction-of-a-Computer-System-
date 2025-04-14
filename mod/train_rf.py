import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RF_MODEL_FILE = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")

# Load data
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

# Drop irrelevant columns
df = df.drop(columns=[col for col in ["battery percentage", "power plugged", "system uptime (hours)"] if col in df.columns], errors="ignore")

# Split features and target
X = df.drop(columns=["total estimated emissions"])
y = df["total estimated emissions"]

# Clean and preprocess the data
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, thresh=int(0.5 * len(X)))  # Drop columns with more than 50% missing
X.fillna(X.median(), inplace=True)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(scaler, SCALER_FILE)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, RF_MODEL_FILE)
print(f" Random Forest model trained and saved to {RF_MODEL_FILE}")

# Evaluate the model
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# Print performance
print("\nRandom Forest Performance:")
print(f"  MAE : {mae_rf:.2f} kg CO2")
print(f"  RMSE: {rmse_rf:.2f} kg CO2")
print(f"  R²  : {r2_rf:.4f}")
