import os
import psutil
import subprocess
import pandas as pd
import time
import requests

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_FILE = os.path.join(DATA_DIR, "system_data.csv")


os.makedirs(DATA_DIR, exist_ok=True)


def get_cpu_power():
    cpu_usage = psutil.cpu_percent(interval=1)
    base_power = 5  
    max_power = 35  
    return base_power + (cpu_usage / 100) * (max_power - base_power)


def get_gpu_power():
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits", shell=True)
        return float(output.decode("utf-8").strip())
    except Exception:
        return 0  


def get_battery_status():
    battery = psutil.sensors_battery()
    if battery:
        return {"Battery Percentage": battery.percent, "Power Plugged": battery.power_plugged}
    return {"Battery Percentage": "N/A", "Power Plugged": "N/A"}  


def get_system_uptime():
    uptime_seconds = time.time() - psutil.boot_time()  # Corrected uptime calculation
    return uptime_seconds / 3600  # seconds to hours


def get_grid_carbon_intensity():
    try:
        response = requests.get("https://api.carbonintensity.org.uk/intensity", timeout=5)
        data = response.json()
        return data["data"][0]["intensity"]["actual"] / 1000  # Convert gCO2/kWh to kgCO2/kWh
    except Exception:
        return 0.5  # UK grid carbon intensity default value


def collect_data(runs=10, interval=10):
    for i in range(runs):
        print(f"Collecting data... ({i+1}/{runs})")
        
        cpu_power = get_cpu_power()
        gpu_power = get_gpu_power()
        uptime_hours = get_system_uptime()
        grid_intensity = get_grid_carbon_intensity()
        battery_status = get_battery_status()

        total_energy_kwh = (cpu_power + gpu_power) * uptime_hours / 1000

        operational_emissions = total_energy_kwh * grid_intensity

        data = {
            "CPU Power (W)": cpu_power,
            "GPU Power (W)": gpu_power,
            "Battery Percentage": battery_status["Battery Percentage"],
            "Power Plugged": battery_status["Power Plugged"],
            "System Uptime (hours)": round(uptime_hours, 2),
            "Energy Use (kWh/year)": round(total_energy_kwh * 24 * 365, 3),  
            "Grid Carbon Intensity (kg CO2/kWh)": grid_intensity,
            "Operational Emissions (kg CO2)": round(operational_emissions, 3),
            "Manufacturing Emissions (kg CO2)": 100,  
            "Disposal Emissions (kg CO2)": 20,  
        }

        df = pd.DataFrame([data])

        file_exists = os.path.exists(DATA_FILE)
        df.to_csv(DATA_FILE, mode='a', header=not file_exists, index=False)

        print(f" Data collected & saved to {DATA_FILE}: {data}")

        time.sleep(interval)  

if __name__ == "__main__":
    collect_data(runs=10, interval=10)
