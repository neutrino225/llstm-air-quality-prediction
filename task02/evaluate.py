import os
import joblib
import sys
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from utils import load_params

matplotlib.use("Agg")


def find_best_model():
    runs = mlflow.search_runs()
    best_run = runs.loc[runs["metrics.mae"].idxmin()]
    return best_run


def load_model(run_id):
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from URI: {model_uri}")
    return mlflow.tensorflow.load_model(model_uri)


def main():
    params = load_params("params.yml")
    features_path = params["data"]["features_save_dir"]

    best_run = find_best_model()
    model = load_model(best_run.run_id)

    # Load the test data
    test_path = os.path.join(features_path, "test.pkl")
    if os.path.exists(test_path):
        X_test, y_test = joblib.load(test_path)
    else:
        print("Error: Test data not found. Ensure featurization.py has been run.")
        sys.exit(1)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")

    # create plots directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Line plot for Actual vs Predicted over time
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="orange", linestyle="dashed")
    plt.title("Actual vs Predicted Over Time")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/actual_vs_predicted.png")

    # Histogram of residuals
    residuals = (y_test - y_pred).flatten()  # Ensure residuals is 1D

    # Line Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="orange", linestyle="dashed")
    plt.title("Actual vs Predicted Over Time")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/line_actual_vs_predicted.png")

    # Histogram of Residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color="green", edgecolor="black", alpha=0.7)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("plots/residual_histogram.png")

    # Cumulative Error Plot
    cumulative_error = abs(residuals).cumsum()
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_error, label="Cumulative Error", color="red")
    plt.title("Cumulative Error Over Time")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Error")
    plt.grid(True)
    plt.savefig("plots/cumulative_error_plot.png")


if __name__ == "__main__":
    main()
