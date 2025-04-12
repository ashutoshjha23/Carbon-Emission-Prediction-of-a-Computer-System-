import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Define file paths
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

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Debug: print columns
print("✅ Cleaned Columns:", df.columns.tolist())

# Check for required target column
if "total estimated emissions" not in df.columns:
    raise KeyError(f"❌ Column 'total estimated emissions' not found. Available columns: {df.columns.tolist()}")

# Drop irrelevant columns if they exist
drop_columns = ["battery percentage", "power plugged", "system uptime (hours)"]
df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")

# Split features and target
X = df.drop(columns=["total estimated emissions"])
y = df["total estimated emissions"]

# Convert all features to numeric and clean missing data
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, thresh=int(0.5 * len(X)))  # Drop columns with >50% missing
X.fillna(X.median(), inplace=True)

# Final NaN check
if X.isnull().sum().sum() > 0:
    raise ValueError("❌ NaN values still exist after preprocessing. Check data!")

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(scaler, SCALER_FILE)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define model with initial hyperparameters
model = XGBRegressor(n_estimators=10, learning_rate=0.05, max_depth=6, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Train the model with the best parameters
best_model.fit(X_train, y_train)

# Evaluate model
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print performance
print(f"\n Model Performance (with Hyperparameter Tuning):")
print(f"   - Mean Absolute Error (MAE): {mae:.2f} kg CO2")
print(f"   - Root Mean Squared Error (RMSE): {rmse:.2f} kg CO2")
print(f"   - R² Score: {r2:.4f}")

# Save model
joblib.dump(best_model, MODEL_FILE)
print(f"\n Model training complete. Saved at: {MODEL_FILE}")
