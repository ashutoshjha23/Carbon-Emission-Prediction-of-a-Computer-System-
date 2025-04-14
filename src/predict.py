import os
import joblib
import pandas as pd
import numpy as np

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_FILE = os.path.join(BASE_DIR, "models", "carbon_footprint_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")
PREDICTION_FILE = os.path.join(BASE_DIR, "data", "predicted_carbon_footprint.csv")

# Load model and scaler
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    raise FileNotFoundError("❌ Model or scaler not found. Train the model first.")
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# Load data
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f" Processed data file not found: {DATA_FILE}. Run preprocess.py first.")
df = pd.read_csv(DATA_FILE)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Drop target column if it exists
if "total estimated emissions" in df.columns:
    df = df.drop(columns=["total estimated emissions"])

# Convert to numeric and handle NaNs
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(axis=1, thresh=int(0.5 * len(df)))  # Drop columns with too many NaNs
df.fillna(df.median(), inplace=True)

# Ensure the features match the ones used for training
scaler_features = scaler.feature_names_in_  # Get the features the scaler was trained on

# Reindex the DataFrame to match the features used in training
df = df.reindex(columns=scaler_features, fill_value=0)

# Scale features
X_scaled = scaler.transform(df)

# Predict
predictions = model.predict(X_scaled)

# Append predictions
df["predicted_emissions (kg co2)"] = predictions

# Save predictions
df.to_csv(PREDICTION_FILE, index=False)
print(f" Predictions saved to: {PREDICTION_FILE}")
