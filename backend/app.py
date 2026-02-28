import os
import sys
import time
import json
import joblib
import psutil
import subprocess
import requests
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor


def compute_mase(y_true, y_pred):
    """Mean Absolute Scaled Error (MASE): MAE / naive-forecast MAE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    naive_mae = np.mean(np.abs(np.diff(y_true)))  # one-step naive forecast
    if naive_mae == 0:
        return 0
    return mae / naive_mae


def compute_bic(y_true, y_pred, n_params):
    """Bayesian Information Criterion approximation for regression."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals ** 2)
    if sse <= 0 or n <= 0:
        return 0
    return n * np.log(sse / n) + n_params * np.log(n)

app = Flask(__name__)
CORS(app)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "carbon_footprint_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
RAW_DATA_FILE = os.path.join(DATA_DIR, "system_data.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_system_data.csv")
PREDICTION_FILE = os.path.join(DATA_DIR, "predicted_carbon_footprint.csv")


def load_model_and_scaler():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return None, None
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler


# ---- Status endpoint ----
@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "model_exists": os.path.exists(MODEL_FILE),
        "scaler_exists": os.path.exists(SCALER_FILE),
        "raw_data_exists": os.path.exists(RAW_DATA_FILE),
        "processed_data_exists": os.path.exists(PROCESSED_DATA_FILE),
        "prediction_exists": os.path.exists(PREDICTION_FILE),
    })


# ---- Step 1: Collect System Data ----
@app.route("/api/collect-data", methods=["POST"])
def collect_data():
    body = request.get_json() or {}
    runs = int(body.get("runs", 5))
    interval = int(body.get("interval", 2))

    collected = []
    for i in range(runs):
        cpu_usage = psutil.cpu_percent(interval=1)
        base_power = 5
        max_power = 35
        cpu_power = base_power + (cpu_usage / 100) * (max_power - base_power)

        try:
            output = subprocess.check_output(
                "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
                shell=True,
            )
            gpu_power = float(output.decode("utf-8").strip())
        except Exception:
            gpu_power = 0

        battery = psutil.sensors_battery()
        battery_pct = battery.percent if battery else "N/A"
        power_plugged = battery.power_plugged if battery else "N/A"
        uptime_hours = (time.time() - psutil.boot_time()) / 3600

        try:
            response = requests.get(
                "https://api.electricitymap.org/v3/carbon-intensity/latest?zone=IN",
                headers={"auth-token": "QYX3Wv8p0oEimgFUScF9"},
                timeout=5,
            )
            grid_intensity = response.json()["carbonIntensity"] / 1000
        except Exception:
            grid_intensity = 0.5

        total_energy_kwh = (cpu_power + gpu_power) * uptime_hours / 1000
        operational_emissions = total_energy_kwh * grid_intensity

        data = {
            "cpu power (w)": round(cpu_power, 2),
            "gpu power (w)": round(gpu_power, 2),
            "battery percentage": battery_pct,
            "power plugged": power_plugged,
            "system uptime (hours)": round(uptime_hours, 2),
            "energy use (kwh/year)": round(total_energy_kwh * 24 * 365, 3),
            "grid carbon intensity (kg co2/kwh)": grid_intensity,
            "operational emissions (kg co2)": round(operational_emissions, 3),
            "manufacturing emissions (kg co2)": 100,
            "disposal emissions (kg co2)": 20,
        }
        collected.append(data)
        if i < runs - 1:
            time.sleep(interval)

    df = pd.DataFrame(collected)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_DATA_FILE, index=False)

    return jsonify({
        "message": f"Collected {len(df)} samples.",
        "data": df.to_dict(orient="records"),
        "columns": df.columns.tolist(),
    })


# ---- Step 1b: Upload CSV ----
@app.route("/api/upload-csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted."}), 400
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {e}"}), 400
    if df.empty:
        return jsonify({"error": "Uploaded CSV is empty."}), 400
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_DATA_FILE, index=False)
    return jsonify({
        "message": f"Uploaded {file.filename} with {len(df)} rows.",
        "data": df.to_dict(orient="records"),
        "columns": df.columns.tolist(),
    })


# ---- Step 2: Preprocess Data ----
@app.route("/api/preprocess", methods=["POST"])
def preprocess_data():
    if not os.path.exists(RAW_DATA_FILE):
        return jsonify({"error": "system_data.csv not found. Collect data first."}), 400

    df = pd.read_csv(RAW_DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    if "power plugged" in df.columns:
        df["power plugged"] = df["power plugged"].astype(str).map({"True": 1, "False": 0})
    df.fillna(df.median(numeric_only=True), inplace=True)

    required_columns = [
        "energy use (kwh/year)",
        "grid carbon intensity (kg co2/kwh)",
        "manufacturing emissions (kg co2)",
        "disposal emissions (kg co2)",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    df["operational emissions"] = df["energy use (kwh/year)"] * df["grid carbon intensity (kg co2/kwh)"]
    df["total estimated emissions"] = (
        df["manufacturing emissions (kg co2)"]
        + df["operational emissions"]
        + df["disposal emissions (kg co2)"]
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    df.to_csv(PROCESSED_DATA_FILE, index=False)

    return jsonify({
        "message": "Preprocessing complete.",
        "data": df.to_dict(orient="records"),
        "columns": df.columns.tolist(),
    })


# ---- Step 3: Train Model ----
@app.route("/api/train", methods=["POST"])
def train_model():
    if not os.path.exists(PROCESSED_DATA_FILE):
        return jsonify({"error": "processed_system_data.csv not found. Preprocess data first."}), 400

    df = pd.read_csv(PROCESSED_DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()

    if "total estimated emissions" not in df.columns:
        return jsonify({"error": "'total estimated emissions' column not found."}), 400

    drop_columns = ["battery percentage", "power plugged", "system uptime (hours)"]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")

    X = df.drop(columns=["total estimated emissions"])
    y = df["total estimated emissions"]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, thresh=int(0.5 * len(X)))
    X.fillna(X.median(), inplace=True)

    if X.isnull().sum().sum() > 0:
        return jsonify({"error": "NaN values still exist after preprocessing."}), 400

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_FILE)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=10, learning_rate=0.05, max_depth=6, random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 6],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid,
        scoring="neg_mean_squared_error", cv=3, verbose=0, n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # percentage
    mase = compute_mase(y_test, y_pred)
    n_params = len(grid_search.best_params_)
    bic = compute_bic(y_test, y_pred, n_params)

    # Sanitize NaN/Inf values (not valid JSON)
    def safe_float(v):
        if v is None or np.isnan(v) or np.isinf(v):
            return 0
        return round(float(v), 4)

    joblib.dump(best_model, MODEL_FILE)

    return jsonify({
        "message": "Model trained and saved.",
        "metrics": {
            "mae": safe_float(mae),
            "rmse": safe_float(rmse),
            "r2": safe_float(r2),
            "mape": safe_float(mape),
            "mase": safe_float(mase),
            "bic": safe_float(bic),
        },
        "best_params": grid_search.best_params_,
    })


# ---- Step 4: Predict ----
@app.route("/api/predict", methods=["POST"])
def predict():
    model, scaler = load_model_and_scaler()
    if model is None:
        return jsonify({"error": "Model or scaler not found. Train the model first."}), 400
    if not os.path.exists(PROCESSED_DATA_FILE):
        return jsonify({"error": "No processed data found. Collect and preprocess data first."}), 400

    df = pd.read_csv(PROCESSED_DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    if "total estimated emissions" in df.columns:
        df = df.drop(columns=["total estimated emissions"])

    scaler_features = scaler.feature_names_in_
    df = df.reindex(columns=scaler_features, fill_value=0)
    X_scaled = scaler.transform(df)
    predictions = model.predict(X_scaled)
    df["predicted_emissions (kg co2)"] = predictions.tolist()
    df.to_csv(PREDICTION_FILE, index=False)

    return jsonify({
        "message": "Prediction complete!",
        "data": df.to_dict(orient="records"),
        "columns": df.columns.tolist(),
    })


# ---- Download predictions CSV ----
@app.route("/api/download-predictions", methods=["GET"])
def download_predictions():
    if not os.path.exists(PREDICTION_FILE):
        return jsonify({"error": "No predictions file found."}), 404
    return send_file(PREDICTION_FILE, as_attachment=True, download_name="predicted_carbon_footprint.csv")


# ---- Visualizations ----
@app.route("/api/visualizations/feature-importance", methods=["GET"])
def feature_importance():
    model, _ = load_model_and_scaler()
    if model is None:
        return jsonify({"error": "Model not found."}), 404
    try:
        importances = model.get_booster().get_score(importance_type="gain")
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        return jsonify({
            "labels": [x[0] for x in sorted_imp],
            "values": [x[1] for x in sorted_imp],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/visualizations/actual-vs-predicted", methods=["GET"])
def actual_vs_predicted():
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not found."}), 404
    if not os.path.exists(PROCESSED_DATA_FILE):
        return jsonify({"error": "Processed data not found."}), 404

    df = pd.read_csv(PROCESSED_DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    if "total estimated emissions" not in df.columns:
        return jsonify({"error": "'total estimated emissions' column not found."}), 400

    X = df.drop(columns=["total estimated emissions"])
    y = df["total estimated emissions"]
    scaler_features = scaler.feature_names_in_
    X = X.reindex(columns=scaler_features, fill_value=0)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    return jsonify({
        "actual": y.tolist(),
        "predicted": y_pred.tolist(),
    })


@app.route("/api/visualizations/error-distribution", methods=["GET"])
def error_distribution():
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not found."}), 404
    if not os.path.exists(PROCESSED_DATA_FILE):
        return jsonify({"error": "Processed data not found."}), 404

    df = pd.read_csv(PROCESSED_DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    if "total estimated emissions" not in df.columns:
        return jsonify({"error": "'total estimated emissions' column not found."}), 400

    X = df.drop(columns=["total estimated emissions"])
    y = df["total estimated emissions"]
    scaler_features = scaler.feature_names_in_
    X = X.reindex(columns=scaler_features, fill_value=0)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    errors = (y - y_pred).tolist()

    return jsonify({"errors": errors})


@app.route("/api/visualizations/model-comparison", methods=["GET"])
def model_comparison():
    if not os.path.exists(PROCESSED_DATA_FILE):
        return jsonify({"error": "Processed data not found."}), 404

    df = pd.read_csv(PROCESSED_DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    drop_columns = ["battery percentage", "power plugged", "system uptime (hours)"]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")

    if "total estimated emissions" not in df.columns:
        return jsonify({"error": "'total estimated emissions' column not found."}), 400

    X = df.drop(columns=["total estimated emissions"])
    y = df["total estimated emissions"]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, thresh=int(0.5 * len(X)))
    X.fillna(X.median(), inplace=True)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model_files = {
        "XGBoost": "carbon_footprint_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "KNN": "knn_model.pkl",
        "SVR": "svr_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
    }
    results = {}
    def safe_float(v):
        if v is None or np.isnan(v) or np.isinf(v):
            return 0
        return round(float(v), 4)

    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            m = joblib.load(path)
            y_pred = m.predict(X_test)
            n_params = X_test.shape[1]  # approximate number of params
            results[name] = {
                "mae": safe_float(mean_absolute_error(y_test, y_pred)),
                "rmse": safe_float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": safe_float(r2_score(y_test, y_pred)),
                "mape": safe_float(mean_absolute_percentage_error(y_test, y_pred) * 100),
                "mase": safe_float(compute_mase(y_test, y_pred)),
                "bic": safe_float(compute_bic(y_test, y_pred, n_params)),
            }

    return jsonify(results)


# ---- Carbon Intensity Visualization ----
@app.route("/api/visualizations/carbon-intensity", methods=["GET"])
def carbon_intensity_viz():
    if not os.path.exists(PROCESSED_DATA_FILE):
        return jsonify({"error": "Processed data not found."}), 404
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df.columns = df.columns.str.strip().str.lower()
    ci_col = "grid carbon intensity (kg co2/kwh)"
    if ci_col not in df.columns:
        return jsonify({"error": f"'{ci_col}' column not found."}), 400
    return jsonify({
        "values": df[ci_col].tolist(),
        "labels": [f"Sample {i+1}" for i in range(len(df))],
        "mean": round(float(df[ci_col].mean()), 6),
        "min": round(float(df[ci_col].min()), 6),
        "max": round(float(df[ci_col].max()), 6),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
