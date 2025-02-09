import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")

## Islamabad
LAT = "33.6995"
LON = "73.0363"

# Bulk data fetch
START_DATE = "2024-12-10"
END_DATE = datetime.now().strftime("%Y-%m-%d")


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


def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


def main():
    start_date = datetime.strptime("2023-01-01", "%Y-%m-%d")

    while True:
        start_date = start_date + timedelta(days=30)
        end_date = start_date + timedelta(days=30)

        data = fetch_pollution_data(
            LAT,
            LON,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            API_KEY,
        )

        print(
            "Passing data from {} to {}".format(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
        )

        # Process and save data
        df = pd.json_normalize(data)

        df["dt"] = pd.to_datetime(df["dt"], unit="s")

        df.columns = [
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

        # Create sequences
        SEQ_LENGTH = 50
        processed_data = df[
            ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
        ].values

        X, y = create_sequences(processed_data[:, 0], SEQ_LENGTH)

        # Send data to the Flask API predict endpoint
        url = "http://localhost:5000/predict"
        response = requests.post(url, json={"input": X.tolist(), "labels": y.tolist()})

        if START_DATE == END_DATE:
            break

        time.sleep(10)


if __name__ == "__main__":
    main()
