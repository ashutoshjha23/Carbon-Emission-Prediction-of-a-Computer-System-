import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data", "system_data.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed_system_data.csv")

if not os.path.exists(DATA_FILE):
    print(f" Error: Data file {DATA_FILE} not found.")
    exit()

try:
    df = pd.read_csv(DATA_FILE)
    print(" Data loaded successfully.")
except Exception as e:
    print(f" Error loading file: {e}")
    exit()

print(" Available Columns:", df.columns.tolist())

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
    print(f" Missing columns: {missing_columns}")
    exit()

df["operational emissions"] = df["energy use (kwh/year)"] * df["grid carbon intensity (kg co2/kwh)"]
df["total estimated emissions"] = df["manufacturing emissions (kg co2)"] + df["operational emissions"] + df["disposal emissions (kg co2)"]

df = df.apply(pd.to_numeric, errors='coerce')

try:
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data Preprocessing Complete. Processed data saved at {OUTPUT_FILE}")
except Exception as e:
    print(f" Error saving file: {e}")
