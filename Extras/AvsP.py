import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load processed data
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")
SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "models", "carbon_footprint_model.pkl")

df = pd.read_csv(DATA_FILE)

# Clean column names (strip spaces and lowercase)
df.columns = df.columns.str.strip().str.lower()

# Drop non-feature columns used during training (if they exist in the test data)
drop_columns = ["battery percentage", "power plugged", "system uptime (hours)"]
df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")

# Check if the 'total estimated emissions' column exists
if "total estimated emissions" in df.columns:
    X = df.drop(columns=["total estimated emissions"])
    y = df["total estimated emissions"]
else:
    raise KeyError("❌ Column 'total estimated emissions' not found in dataset.")

# Load scaler and model
scaler = joblib.load(SCALER_FILE)
model = joblib.load(MODEL_FILE)

# Print the features the scaler expects (features used during training)
print("Features used for training:")
print(scaler.feature_names_in_)

# Ensure the columns in X match those used during training
scaler_features = scaler.feature_names_in_
X = X.reindex(columns=scaler_features, fill_value=0)

# Scale data
X_scaled = scaler.transform(X)

# Split data (for evaluation consistency)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Predict
y_pred = model.predict(X_test)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Emissions (kg CO₂)")
plt.ylabel("Predicted Emissions (kg CO₂)")
plt.title("Actual vs Predicted Emissions")
plt.grid(True)
plt.tight_layout()
plt.show()
