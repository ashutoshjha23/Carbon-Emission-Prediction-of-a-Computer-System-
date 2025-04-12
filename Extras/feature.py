import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import plot_importance

# Load model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_FILE = os.path.join(BASE_DIR, "models", "carbon_footprint_model.pkl")

model = joblib.load(MODEL_FILE)

# Plot feature importance
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='gain', max_num_features=10, height=0.5)
plt.title("Top 10 Feature Importances (by Gain)")
plt.tight_layout()
plt.show()
