import os
import joblib
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_FILE = os.path.join(BASE_DIR, "models", "carbon_footprint_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")  

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    raise FileNotFoundError("❌ Model or scaler not found. Train the model first.")
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"❌ Processed data file not found: {DATA_FILE}. Run preprocess.py first.")
df = pd.read_csv(DATA_FILE)
if "total estimated emissions" in df.columns:
    df = df.drop(columns=["total estimated emissions"])

df = df.apply(pd.to_numeric, errors="coerce")

scaler_features = scaler.feature_names_in_  
df = df.reindex(columns=scaler_features, fill_value=0)  

# Scale features
scaled_data = scaler.transform(df)

# Make predictions
predictions = model.predict(scaled_data)

# Add predictions to dataframe
df["predicted_emissions (kg CO2)"] = predictions

# saving predictions
PREDICTION_FILE = os.path.join(BASE_DIR, "data", "predicted_carbon_footprint.csv")
df.to_csv(PREDICTION_FILE, index=False)

print(f" Predictions saved to {PREDICTION_FILE}")
