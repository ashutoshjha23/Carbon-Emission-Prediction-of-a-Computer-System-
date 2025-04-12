import os
import psutil
import subprocess
import pandas as pd
import time
import requests

# ✅ Define Data Storage Path
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_FILE = os.path.join(DATA_DIR, "system_data.csv")

# ✅ Ensure the 'data' directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Get Real-Time CPU Power Consumption
def get_cpu_power():
    cpu_usage = psutil.cpu_percent(interval=1)
    base_power = 5   # Idle CPU power in watts
    max_power = 35   # Max CPU power in watts
    return base_power + (cpu_usage / 100) * (max_power - base_power)

# ✅ Get Real-Time GPU Power Consumption (NVIDIA Only)
def get_gpu_power():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
            shell=True
        )
        return float(output.decode("utf-8").strip())
    except Exception:
        return 0  # If no GPU is detected

#  Get Battery Power (Laptops Only)
def get_battery_status():
    battery = psutil.sensors_battery()
    if battery:
        return {"Battery Percentage": battery.percent, "Power Plugged": battery.power_plugged}
    return {"Battery Percentage": "N/A", "Power Plugged": "N/A"}

#  Get System Uptime in Hours
def get_system_uptime():
    uptime_seconds = time.time() - psutil.boot_time()
    return uptime_seconds / 3600  # Convert seconds to hours

#  Get Grid Carbon Intensity (Electricity Map API for India)
def get_grid_carbon_intensity():
    try:
        response = requests.get(
            "https://api.electricitymap.org/v3/carbon-intensity/latest?zone=IN",
            headers={"auth-token": "QYX3Wv8p0oEimgFUScF9"},
            timeout=5
        )
        data = response.json()
        return data["carbonIntensity"] / 1000  # Convert gCO2/kWh to kgCO2/kWh
    except Exception:
        return 0.5  # Default value if API fails

#  Function to Collect System Data
def collect_data(runs=10, interval=10):
    for i in range(runs):
        print(f"🔄 Collecting data... ({i+1}/{runs})")
        
        cpu_power = get_cpu_power()
        gpu_power = get_gpu_power()
        uptime_hours = get_system_uptime()
        grid_intensity = get_grid_carbon_intensity()
        battery_status = get_battery_status()

        #  Calculate Total Energy Use in kWh
        total_energy_kwh = (cpu_power + gpu_power) * uptime_hours / 1000

        # Calculate Operational Emissions (kg CO₂)
        operational_emissions = total_energy_kwh * grid_intensity

        #  Gather Data
        data = {
            "CPU Power (W)": cpu_power,
            "GPU Power (W)": gpu_power,
            "Battery Percentage": battery_status["Battery Percentage"],
            "Power Plugged": battery_status["Power Plugged"],
            "System Uptime (hours)": round(uptime_hours, 2),
            "Energy Use (kWh/year)": round(total_energy_kwh * 24 * 365, 3),
            "Grid Carbon Intensity (kg CO2/kWh)": grid_intensity,
            "Operational Emissions (kg CO2)": round(operational_emissions, 3),
            "Manufacturing Emissions (kg CO2)": 100,  # Placeholder
            "Disposal Emissions (kg CO2)": 20,       # Placeholder
        }

        df = pd.DataFrame([data])

        #  Append data to CSV file
        file_exists = os.path.exists(DATA_FILE)
        df.to_csv(DATA_FILE, mode='a', header=not file_exists, index=False)

        print(f" Data collected & saved to {DATA_FILE}: {data}")
        time.sleep(interval)

#  Run Data Collection
if __name__ == "__main__":
    collect_data(runs=550, interval=2)
