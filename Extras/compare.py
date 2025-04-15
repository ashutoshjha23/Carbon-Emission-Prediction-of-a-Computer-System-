import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()
df = df.drop(columns=[col for col in ["battery percentage", "power plugged", "system uptime (hours)"] if col in df.columns], errors="ignore")

X = df.drop(columns=["total estimated emissions"])
y = df["total estimated emissions"]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, thresh=int(0.5 * len(X)))
X.fillna(X.median(), inplace=True)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

model_files = {
    "XGBoost": "carbon_footprint_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "KNN": "knn_model.pkl",
    "SVR": "svr_model.pkl",
    "Decision Tree": "decision_tree_model.pkl"
}

# === Load Models & Evaluate ===
results = {}

for model_name, file_name in model_files.items():
    model_path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Loaded {model_name} model")
        y_pred = model.predict(X_test)
        results[model_name] = evaluate(y_test, y_pred)
    else:
        print(f"❌ {model_name} model not found at {model_path}")

for name, metrics in results.items():
    print(f"\n{name} Performance:")
    print(f"  MAE : {metrics['MAE']:.2f} kg CO2")
    print(f"  RMSE: {metrics['RMSE']:.2f} kg CO2")
    print(f"  R²  : {metrics['R2']:.4f}")

labels = list(results.keys())
mae_vals = [results[m]["MAE"] for m in labels]
rmse_vals = [results[m]["RMSE"] for m in labels]
r2_vals = [results[m]["R2"] for m in labels]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, mae_vals, width, label='MAE', color='coral')
ax.bar(x, rmse_vals, width, label='RMSE', color='deepskyblue')
ax.bar(x + width, r2_vals, width, label='R² Score', color='limegreen')

ax.set_ylabel('Error / Score')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, axis='y')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(labels, mae_vals, marker='o', label='MAE', color='coral')
ax.plot(labels, rmse_vals, marker='s', label='RMSE', color='deepskyblue')
ax.plot(labels, r2_vals, marker='^', label='R² Score', color='limegreen')

ax.set_title('Model Performance Comparison (Line Plot)')
ax.set_ylabel('Error / Score')
ax.set_xlabel('Models')
ax.grid(True)
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
