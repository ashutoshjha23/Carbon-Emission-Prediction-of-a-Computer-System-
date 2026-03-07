import matplotlib.pyplot as plt
import numpy as np

# ---- Sample data (replace with your actual values) ----
models = ["Decision Tree", "KNN", "Random Forest", "SVR", "XGBoost"]

metrics = {
    "MAE":      [0.12, 0.85, 0.05, 1.23, 0.01],
    "RMSE":     [0.18, 1.02, 0.08, 1.45, 0.02],
    "R²":       [-49800, -60200, -45300, -27500, -1200],
    "MAPE (%)": [0.10, 0.71, 0.04, 1.02, 0.01],
    "MASE":     [0.15, 0.90, 0.06, 1.30, 0.02],
    "BIC":      [12.5, 18.3, 10.1, 22.7, 5.8],
}

colors = {
    "MAE": "#43cea2",
    "RMSE": "#185a9d",
    "R²": "#fbbf24",
    "MAPE (%)": "#ef4444",
    "MASE": "#8b5cf6",
    "BIC": "#f97316",
}

hints = {
    "MAE": "lower is better",
    "RMSE": "lower is better",
    "R²": "closer to 1 is better",
    "MAPE (%)": "lower is better",
    "MASE": "lower is better",
    "BIC": "lower is better",
}

# ---- One chart per metric ----
x = np.arange(len(models))

for metric, values in metrics.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, values, color=colors[metric], edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Model Comparison — {metric} ({hints[metric]})", fontsize=14, color="#185a9d")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# ---- Carbon Intensity ----
ci_values = [0.508, 0.508, 0.508, 0.508, 0.508]  # replace with your data
ci_labels = [f"Sample {i+1}" for i in range(len(ci_values))]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ci_labels, ci_values, marker="o", color="#10b981", linewidth=2, markersize=6)
ax.fill_between(ci_labels, ci_values, alpha=0.2, color="#10b981")
ax.set_ylabel("kg CO₂/kWh", fontsize=12)
ax.set_title("Grid Carbon Intensity per Sample", fontsize=14, color="#185a9d")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()