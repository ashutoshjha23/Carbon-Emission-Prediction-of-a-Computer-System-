import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

#  Define file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "carbon_footprint_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"❌ Processed dataset not found: {DATA_FILE}. Run preprocess.py first.")

# Load data
df = pd.read_csv(DATA_FILE)
print(f" Data Shape: {df.shape}")  # Debugging dataset size

if "total estimated emissions" not in df.columns:
    raise KeyError("❌ Column 'total estimated emissions' not found in dataset. Check preprocess.py.")

drop_columns = ["Battery Percentage", "Power Plugged", "System Uptime (hours)"]
df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")

X = df.drop(columns=["total estimated emissions"])
y = df["total estimated emissions"]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, thresh=int(0.5 * len(X)))
X.fillna(X.median(), inplace=True)


if X.isnull().sum().sum() > 0:
    raise ValueError("❌ NaN values still exist after preprocessing. Check data!")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(scaler, SCALER_FILE)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f" Model Performance:")
print(f"   - Mean Absolute Error (MAE): {mae:.2f} kg CO2")
print(f"   - Root Mean Squared Error (RMSE): {rmse:.2f} kg CO2")
print(f"   - R² Score: {r2:.4f}")

# Save the trained model
joblib.dump(model, MODEL_FILE)
print(f" Model training complete. Saved at {MODEL_FILE}")
