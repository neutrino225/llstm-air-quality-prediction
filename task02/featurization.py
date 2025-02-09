import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys
from utils import load_params

# Load parameters from params.yml
params = load_params("params.yml")
SEQ_LENGTH = params["training"]["seq_length"]


def process_data(df):
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df.fillna(method="ffill", inplace=True)

    # Remove outliers using the IQR method
    for column in ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.where(
            (df[column] < lower_bound) | (df[column] > upper_bound),
            np.nan,
            df[column],
        )

    # Replace outliers with median
    for column in ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]:
        df[column].fillna(df[column].median(), inplace=True)

    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(
        df[["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]]
    )
    return data_scaled


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
    data_path = params["data"]["path"]
    features_output_directory = params["data"]["features_save_dir"]

    os.makedirs(features_output_directory, exist_ok=True)

    TRAIN_SAVE_PATH = os.path.join(features_output_directory, "train.pkl")
    TEST_SAVE_PATH = os.path.join(features_output_directory, "test.pkl")

    # Load data
    df = pd.read_csv(data_path)

    # Process data
    processed_data = process_data(df)

    # Create sequences
    X, y = create_sequences(processed_data[:, 0], SEQ_LENGTH)

    # Train-test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Save data as pkl
    os.makedirs("data/features", exist_ok=True)

    joblib.dump((X_train, y_train), TRAIN_SAVE_PATH)
    joblib.dump((X_test, y_test), TEST_SAVE_PATH)


if __name__ == "__main__":
    main()
