import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import ParameterGrid
import joblib
import os
from utils import load_params

# Load parameters from config file
params = load_params("params.yml")
SEQ_LENGTH = params["training"]["seq_length"]
BATCH_SIZE = params["training"]["batch_size"]
NUM_EPOCHS = params["training"]["num_epochs"]


# Function to create the LSTM model
def create_model(optimizer="adam", units=64):
    model = Sequential(
        [
            LSTM(units, input_shape=(SEQ_LENGTH, 1), return_sequences=True),
            LSTM(units // 2, return_sequences=False),
            Dense(1),
        ]
    )
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def main():
    features_path = params["data"]["features_save_dir"]

    if not os.path.exists(features_path):
        print(
            f"Error: Features path '{features_path}' does not exist. Run featurization.py first."
        )
        return

    # Load the train/test data
    train_path = os.path.join(features_path, "train.pkl")
    test_path = os.path.join(features_path, "test.pkl")

    if os.path.exists(train_path) and os.path.exists(test_path):
        X_train, y_train = joblib.load(train_path)
        X_test, y_test = joblib.load(test_path)
    else:
        print(
            "Error: Train or test data not found. Ensure featurization.py has been run."
        )
        return

    # Hyperparameter grid to search
    param_grid = {
        "optimizer": ["adam", "rmsprop"],
        "units": [64, 128],
        "batch_size": [16, 32],
    }

    best_model = None
    best_mae = float("inf")
    best_mse = float("inf")

    # Loop through hyperparameters manually
    # Track and save all the models in MLflow
    for params_combination in ParameterGrid(param_grid):
        optimizer = params_combination["optimizer"]
        units = params_combination["units"]
        batch_size = params_combination["batch_size"]

        print(f"Training model with: {params_combination}")

        # Start MLflow run
        with mlflow.start_run():
            mlflow.log_params(params_combination)

            mlflow.tensorflow.autolog()

            # Create and train the model
            model = create_model(optimizer=optimizer, units=units)

            # Train the model
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
            )

            # Evaluate the model
            mse, mae = model.evaluate(X_test, y_test, verbose=0)

            if mae < best_mae:
                best_mae = mae
                best_mse = mse
                best_model = model

            # Log metrics
            mlflow.log_metric("final_mae", mae)
            mlflow.log_metric("final_mse", mse)

    print(f"Best model has MAE: {best_mae} and MSE: {best_mse}")
    mlflow.tensorflow.mlflow.log_model(best_model, "best_model")

    if not os.path.exists("models"):
        os.makedirs("models")

    best_model.save("models/best_model.h5")


if __name__ == "__main__":
    main()
