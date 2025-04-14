import os
import pandas as pd
import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SVR_MODEL_FILE = os.path.join(MODEL_DIR, "svr_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")

# Load data
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

# Drop irrelevant columns
df = df.drop(columns=[col for col in ["battery percentage", "power plugged", "system uptime (hours)"] if col in df.columns], errors="ignore")

# Features and target
X = df.drop(columns=["total estimated emissions"])
y = df["total estimated emissions"]

# Clean and scale data
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, thresh=int(0.5 * len(X)))
X.fillna(X.median(), inplace=True)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(scaler, SCALER_FILE)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVR model
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
svr_model.fit(X_train, y_train)

# Save model
joblib.dump(svr_model, SVR_MODEL_FILE)
print(f" SVR model trained and saved to {SVR_MODEL_FILE}")

# Evaluate
y_pred_svr = svr_model.predict(X_test)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)

print("\nSupport Vector Regressor Performance:")
print(f"  MAE : {mae_svr:.2f} kg CO2")
print(f"  RMSE: {rmse_svr:.2f} kg CO2")
print(f"  R²  : {r2_svr:.4f}")
