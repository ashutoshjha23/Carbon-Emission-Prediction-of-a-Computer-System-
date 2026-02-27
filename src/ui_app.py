import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_FILE = os.path.join(BASE_DIR, "models", "carbon_footprint_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")
PREDICTION_FILE = os.path.join(BASE_DIR, "data", "predicted_carbon_footprint.csv")


st.set_page_config(page_title="Carbon Emission Prediction", layout="wide", page_icon="🌱")
st.markdown("""
<style>
.main {
    background-color: #f7f9fa;
}
.stButton>button {
    color: white;
    background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
    border-radius: 8px;
    font-weight: bold;
}
.stDownloadButton>button {
    color: white;
    background: #185a9d;
    border-radius: 8px;
    font-weight: bold;
}
.stDataFrame {
    background: #fff;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='color:#185a9d;font-size:2.5rem;font-weight:700;'>💻 Carbon Emission Prediction Dashboard</h1>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2909/2909765.png", width=80)
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Predict Emissions", "Visualizations", "About"],
    format_func=lambda x: {"Predict Emissions": "🟢 Full Workflow", "Visualizations": "📊 Visualizations", "About": "ℹ️ About"}[x])

# --- Load Model and Scaler ---
def load_model_and_scaler():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        st.error("Model or scaler not found. Please train the model first.")
        return None, None
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

# --- Predict Emissions Page ---


if page == "Predict Emissions":
    st.header("Full Workflow: Data Collection → Preprocessing → Training → Prediction")
    import psutil
    import subprocess
    import requests
    import time
    import joblib
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from xgboost import XGBRegressor
    # --- Data Collection ---
    st.subheader("Step 1: Collect Real-Time System Data")
    runs = st.number_input("Number of samples (runs)", min_value=1, max_value=100, value=5)
    interval = st.number_input("Interval between samples (seconds)", min_value=1, max_value=60, value=2)
    collect_btn = st.button("Collect Data")
    if collect_btn:
        collected = []
        progress = st.progress(0)
        for i in range(int(runs)):
            cpu_usage = psutil.cpu_percent(interval=1)
            base_power = 5
            max_power = 35
            cpu_power = base_power + (cpu_usage / 100) * (max_power - base_power)
            try:
                output = subprocess.check_output(
                    "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
                    shell=True
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
                    timeout=5
                )
                grid_intensity = response.json()["carbonIntensity"] / 1000
            except Exception:
                grid_intensity = 0.5
            total_energy_kwh = (cpu_power + gpu_power) * uptime_hours / 1000
            operational_emissions = total_energy_kwh * grid_intensity
            data = {
                "cpu power (w)": cpu_power,
                "gpu power (w)": gpu_power,
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
            progress.progress((i+1)/runs)
            time.sleep(interval)
        df = pd.DataFrame(collected)
        st.success(f"Collected {len(df)} samples.")
        st.dataframe(df)
        # Save to system_data.csv
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(os.path.join(data_dir, "system_data.csv"), index=False)
    # --- Preprocessing ---
    st.markdown("---")
    st.subheader("Step 2: Preprocess Data")
    if st.button("Preprocess Data"):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        data_file = os.path.join(data_dir, "system_data.csv")
        output_file = os.path.join(data_dir, "processed_system_data.csv")
        if not os.path.exists(data_file):
            st.error("system_data.csv not found. Collect data first.")
        else:
            df = pd.read_csv(data_file)
            df.columns = df.columns.str.strip().str.lower()
            if "power plugged" in df.columns:
                df["power plugged"] = df["power plugged"].astype(str).map({"True": 1, "False": 0})
            df.fillna(df.median(), inplace=True)
            required_columns = [
                "energy use (kwh/year)", "grid carbon intensity (kg co2/kwh)",
                "manufacturing emissions (kg co2)", "disposal emissions (kg co2)"
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing columns: {missing_columns}")
            else:
                df["operational emissions"] = df["energy use (kwh/year)"] * df["grid carbon intensity (kg co2/kwh)"]
                df["total estimated emissions"] = df["manufacturing emissions (kg co2)"] + df["operational emissions"] + df["disposal emissions (kg co2)"]
                df = df.apply(pd.to_numeric, errors='coerce')
                df.to_csv(output_file, index=False)
                st.success(f"Preprocessing complete. Saved to processed_system_data.csv.")
                st.dataframe(df)
    # --- Model Training ---
    st.markdown("---")
    st.subheader("Step 3: Train Model")
    if st.button("Train Model"):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        data_file = os.path.join(data_dir, "processed_system_data.csv")
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        model_file = os.path.join(model_dir, "carbon_footprint_model.pkl")
        scaler_file = os.path.join(model_dir, "scaler.pkl")
        if not os.path.exists(data_file):
            st.error("processed_system_data.csv not found. Preprocess data first.")
        else:
            df = pd.read_csv(data_file)
            df.columns = df.columns.str.strip().str.lower()
            if "total estimated emissions" not in df.columns:
                st.error("'total estimated emissions' column not found.")
            else:
                drop_columns = ["battery percentage", "power plugged", "system uptime (hours)"]
                df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")
                X = df.drop(columns=["total estimated emissions"])
                y = df["total estimated emissions"]
                X = X.apply(pd.to_numeric, errors="coerce")
                X = X.dropna(axis=1, thresh=int(0.5 * len(X)))
                X.fillna(X.median(), inplace=True)
                if X.isnull().sum().sum() > 0:
                    st.error("NaN values still exist after preprocessing. Check data!")
                else:
                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(X)
                    os.makedirs(model_dir, exist_ok=True)
                    joblib.dump(scaler, scaler_file)
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                    model = XGBRegressor(n_estimators=10, learning_rate=0.05, max_depth=6, random_state=42)
                    param_grid = {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.05],
                        'max_depth': [3, 6],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_model.fit(X_train, y_train)
                    y_pred = best_model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    joblib.dump(best_model, model_file)
                    st.success(f"Model trained and saved. MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    # --- Prediction ---
    st.markdown("---")
    st.subheader("Step 4: Predict Emissions (on Processed Data)")
    model, scaler = load_model_and_scaler()
    if model is not None and scaler is not None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        data_file = os.path.join(data_dir, "processed_system_data.csv")
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            df.columns = df.columns.str.strip().str.lower()
            if "total estimated emissions" in df.columns:
                df = df.drop(columns=["total estimated emissions"])
            scaler_features = scaler.feature_names_in_
            df = df.reindex(columns=scaler_features, fill_value=0)
            X_scaled = scaler.transform(df)
            predictions = model.predict(X_scaled)
            df["predicted_emissions (kg co2)"] = predictions
            st.success("Prediction complete!")
            st.dataframe(df)
            st.download_button(
                label="Download Predictions as CSV",
                data=df.to_csv(index=False).encode(),
                file_name="predicted_carbon_footprint.csv",
                mime="text/csv"
            )
        else:
            st.info("No processed data found. Collect and preprocess data first.")


# --- Visualizations Page ---
elif page == "Visualizations":
    st.header("📊 Model Visualizations & Insights")
    model, scaler = load_model_and_scaler()
    import matplotlib.pyplot as plt
    import numpy as np
    import joblib
    import os
    import pandas as pd
    st.markdown("---")
    # Feature Importance
    st.subheader("Feature Importance (XGBoost)")
    if model is not None:
        try:
            from xgboost import plot_importance
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_importance(model, importance_type='gain', max_num_features=10, height=0.5, ax=ax)
            ax.set_title("Top 10 Feature Importances (by Gain)", fontsize=16, color="#185a9d")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Feature importance plot not available: {e}")
    st.markdown("---")
    # Actual vs Predicted
    st.subheader("Actual vs Predicted (Example Data)")
    if scaler is not None and os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df.columns = df.columns.str.strip().str.lower()
        if "total estimated emissions" in df.columns:
            X = df.drop(columns=["total estimated emissions"])
            y = df["total estimated emissions"]
            scaler_features = scaler.feature_names_in_
            X = X.reindex(columns=scaler_features, fill_value=0)
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.scatter(y, y_pred, alpha=0.6, color='#43cea2', edgecolor='k')
            ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
            ax2.set_xlabel("Actual Emissions (kg CO₂)", fontsize=12)
            ax2.set_ylabel("Predicted Emissions (kg CO₂)", fontsize=12)
            ax2.set_title("Actual vs Predicted Emissions", fontsize=15, color="#185a9d")
            ax2.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig2)
            # Error Distribution
            st.subheader("Prediction Error Distribution")
            errors = y - y_pred
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.hist(errors, bins=20, color='#185a9d', alpha=0.7, edgecolor='white')
            ax3.set_xlabel("Prediction Error (kg CO₂)")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Distribution of Prediction Errors", fontsize=14)
            st.pyplot(fig3)
    st.markdown("---")
    # Model Comparison
    st.subheader("Model Comparison (MAE, RMSE, R²)")
    # Try to load and compare all models
    model_files = {
        "XGBoost": "carbon_footprint_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "KNN": "knn_model.pkl",
        "SVR": "svr_model.pkl",
        "Decision Tree": "decision_tree_model.pkl"
    }
    results = {}
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df.columns = df.columns.str.strip().str.lower()
        drop_columns = ["battery percentage", "power plugged", "system uptime (hours)"]
        df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")
        if "total estimated emissions" in df.columns:
            X = df.drop(columns=["total estimated emissions"])
            y = df["total estimated emissions"]
            from sklearn.preprocessing import MinMaxScaler
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.dropna(axis=1, thresh=int(0.5 * len(X)))
            X.fillna(X.median(), inplace=True)
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            for model_name, file_name in model_files.items():
                model_path = os.path.join(os.path.dirname(__file__), "..", "models", file_name)
                if os.path.exists(model_path):
                    m = joblib.load(model_path)
                    y_pred = m.predict(X_test)
                    results[model_name] = {
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "R2": r2_score(y_test, y_pred)
                    }
            if results:
                labels = list(results.keys())
                mae_vals = [results[m]["MAE"] for m in labels]
                rmse_vals = [results[m]["RMSE"] for m in labels]
                r2_vals = [results[m]["R2"] for m in labels]
                x = np.arange(len(labels))
                width = 0.25
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                ax4.bar(x - width, mae_vals, width, label='MAE', color='#43cea2')
                ax4.bar(x, rmse_vals, width, label='RMSE', color='#185a9d')
                ax4.bar(x + width, r2_vals, width, label='R² Score', color='gold')
                ax4.set_ylabel('Error / Score')
                ax4.set_title('Model Comparison', fontsize=15, color="#185a9d")
                ax4.set_xticks(x)
                ax4.set_xticklabels(labels)
                ax4.legend()
                ax4.grid(True, axis='y', linestyle='--', alpha=0.5)
                st.pyplot(fig4)
    st.markdown("---")

# --- About Page ---
else:
    st.header("ℹ️ About This Project")
    st.markdown("""
    <div style='font-size:1.1rem;'>
    <b>Carbon Emission Prediction of a Computer System</b> is a professional dashboard for collecting, processing, and analyzing the carbon footprint of your computing device using machine learning.<br><br>
    <b>Features:</b>
    <ul>
    <li>Collect real-time system data with customizable intervals and sample size</li>
    <li>Preprocess and clean your data with one click</li>
    <li>Train advanced ML models (XGBoost, Random Forest, etc.) and compare their performance</li>
    <li>Predict emissions and download results</li>
    <li>Visualize feature importance, error distributions, and model comparisons</li>
    </ul>
    <b>Developed by:</b> <a href="https://github.com/ashutoshjha23" target="_blank">ashutoshjha23</a><br>
    <b>Powered by:</b> Streamlit, scikit-learn, XGBoost, Python
    </div>
    """, unsafe_allow_html=True)
