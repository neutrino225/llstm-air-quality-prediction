import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils import load_params

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")

## Islamabad
LAT = "33.6995"
LON = "73.0363"

# Bulk data fetch for model training
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"


def fetch_pollution_data(lat, lon, start_date, end_date, api_key):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    data = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    while current_date <= end_date:
        timestamp = int(current_date.timestamp())
        response = requests.get(
            url,
            params={
                "lat": lat,
                "lon": lon,
                "start": timestamp,
                "end": timestamp + 86400,  # Fetch one day's data
                "appid": api_key,
            },
        )
        if response.status_code == 200:
            data.extend(response.json().get("list", []))
        else:
            print(f"Error: {response.status_code}, {response.text}")
        current_date += timedelta(days=1)

    return data


def main():
    data = fetch_pollution_data(LAT, LON, START_DATE, END_DATE, API_KEY)

    # Load parameters from params.yml
    params = load_params("params.yml")

    data_save_path = params["data"]["path"]

    # Process and save data
    df = pd.json_normalize(data)
    df["dt"] = pd.to_datetime(df["dt"], unit="s")
    df.columns = ["date", "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    df.to_csv(data_save_path, index=False)

    print(f"Data saved to {data_save_path}")


if __name__ == "__main__":
    main()
