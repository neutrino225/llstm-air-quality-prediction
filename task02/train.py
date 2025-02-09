import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import sys
import joblib
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
from utils import load_params
import json

import matplotlib

matplotlib.use("Agg")

params = load_params("params.yml")
SEQ_LENGTH = params["training"]["seq_length"]
BATCH_SIZE = params["training"]["batch_size"]
NUM_EPOCHS = params["training"]["num_epochs"]

# Suppress tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    features_path = params["data"]["features_save_dir"]

    # Check if features path exists
    if not os.path.exists(features_path):
        print(
            f"Error: Features path '{features_path}' does not exist. Run featurization.py first."
        )
        sys.exit(1)

    # Paths for train and test data
    train_path = os.path.join(features_path, "train.pkl")
    test_path = os.path.join(features_path, "test.pkl")

    if os.path.exists(train_path) and os.path.exists(test_path):
        X_train, y_train = joblib.load(train_path)
        X_test, y_test = joblib.load(test_path)
    else:
        print(
            "Error: Train or test data not found. Ensure featurization.py has been run."
        )
        sys.exit(1)

    model = Sequential(
        [
            LSTM(64, input_shape=(SEQ_LENGTH, 1), return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    with mlflow.start_run():
        mlflow.tensorflow.autolog()

        # Log hyperparameters
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss", "mse")
        mlflow.log_param("metrics", "mae")
        mlflow.log_param("sequence_length", SEQ_LENGTH)
        mlflow.log_param("epochs", NUM_EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)

        # Train the model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
        )

        # # Log the model
        # mlflow.tensorflow.log_model(
        #     model=model,
        #     artifact_path="model",
        #     registered_model_name="AQI_LSTM",
        #     conda_env="environment.yml",
        #     code_paths=["src/fetch_data.py", "src/featurization.py", "src/train.py"],
        #     input_example=X_test[0:1],
        # )

        ## Save final metrics in metrics.json
        metrics = {metric: values[-1] for metric, values in history.history.items()}

        with open("metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    main()
