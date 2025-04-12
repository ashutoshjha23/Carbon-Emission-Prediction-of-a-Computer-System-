import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define file paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

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

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Load Pre-trained XGBoost Model ===
xgb_model = joblib.load(os.path.join(MODEL_DIR, "carbon_footprint_model.pkl"))
print("✅ Loaded pre-trained XGBoost model")

# Predict with XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# === Load Trained Random Forest Model ===
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
print("✅ Loaded trained Random Forest model")

# Predict with Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluation function
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

# Get results
results = {
    "XGBoost": evaluate(y_test, y_pred_xgb),
    "Random Forest": evaluate(y_test, y_pred_rf)
}

# Print results
for name, metrics in results.items():
    print(f"\n{name} Performance:")
    print(f"  MAE : {metrics['MAE']:.2f} kg CO2")
    print(f"  RMSE: {metrics['RMSE']:.2f} kg CO2")
    print(f"  R²  : {metrics['R2']:.4f}")

# Plot comparison
labels = list(results.keys())
mae_vals = [results[m]["MAE"] for m in labels]
rmse_vals = [results[m]["RMSE"] for m in labels]
r2_vals = [results[m]["R2"] for m in labels]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width, mae_vals, width, label='MAE', color='coral')
ax.bar(x, rmse_vals, width, label='RMSE', color='deepskyblue')
ax.bar(x + width, r2_vals, width, label='R² Score', color='limegreen')

ax.set_ylabel('Error / Score')
ax.set_title('Model Comparison: XGBoost vs Random Forest')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, axis='y')

plt.tight_layout()
plt.show()
