import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import schedule
import time
import subprocess

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv("WEATHER_API_KEY")
LAT = "33.6995"  # Islamabad latitude
LON = "73.0363"  # Islamabad longitude
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "live_data.csv")


def fetch_live_data(lat, lon, api_key):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
    response = requests.get(
        url,
        params={
            "lat": lat,
            "lon": lon,
            "appid": api_key,
        },
    )
    if response.status_code == 200:
        data = response.json().get("list", [])
        return data
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def save_live_data():
    print(
        f"[{datetime.now()}] Fetching live data... at {LAT}, {LON} --> {datetime.now()}"
    )
    live_data = fetch_live_data(LAT, LON, API_KEY)

    if live_data:
        df_live = pd.json_normalize(live_data)
        df_live["dt"] = pd.to_datetime(df_live["dt"], unit="s")
        df_live.columns = [
            "date",
            "aqi",
            "co",
            "no",
            "no2",
            "o3",
            "so2",
            "pm2_5",
            "pm10",
            "nh3",
        ]

        os.makedirs(DATA_DIR, exist_ok=True)

        # Save to CSV
        if os.path.exists(DATA_FILE):
            df_live.to_csv(DATA_FILE, mode="a", header=False, index=False)
        else:
            df_live.to_csv(DATA_FILE, index=False)

        print(f"[{datetime.now()}] Data saved to {DATA_FILE}")

        version_data()
    else:
        print(f"[{datetime.now()}] No data fetched.")


# Version control with DVC
def version_data():
    try:
        print(f"[{datetime.now()}] Updating DVC repository...")
        subprocess.run(["dvc", "add", DATA_FILE], check=True)
        subprocess.run(["dvc", "commit"], check=True)
        subprocess.run(["dvc", "push"], check=True)
        print(f"[{datetime.now()}] DVC repository updated.")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] Error during DVC operations: {e}")


# Schedule the data collection
def schedule_data_collection():
    schedule.every(0.05).hours.do(
        save_live_data
    )  ## Fetches data every 3 mins for demo purposes

    print("Starting scheduled data collection...")
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    schedule_data_collection()
