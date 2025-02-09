import mlflow
import mlflow.tensorflow
import os
import pandas as pd
from flask import Flask, request, jsonify, Response
from prettytable import PrettyTable
from prometheus_client import Counter, Gauge, Summary, generate_latest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import psutil

# Define Prometheus metrics
prediction_counter = Counter("total_predictions", "Total number of predictions made")
prediction_latency = Summary(
    "prediction_latency_seconds", "Prediction latency in seconds"
)
prediction_errors = Counter("prediction_errors", "Total number of prediction errors")
mae_metric = Gauge("model_mae", "Mean Absolute Error of predictions")
mse_metric = Gauge("model_mse", "Mean Squared Error of predictions")
r2_metric = Gauge("model_r2_score", "R2 Score of predictions")

# Hardware usage metrics
cpu_usage = Gauge("cpu_usage_percent", "CPU Usage in percentage")
ram_usage = Gauge("ram_usage_percent", "RAM Usage in percentage")
disk_usage = Gauge("disk_usage_percent", "Disk Usage in percentage")
gpu_usage = Gauge("gpu_usage_percent", "GPU Usage in percentage (if available)")


def update_hardware_metrics():
    """Update hardware metrics for Prometheus."""
    cpu_usage.set(psutil.cpu_percent())
    ram_usage.set(psutil.virtual_memory().percent)
    disk_usage.set(psutil.disk_usage("/").percent)

    try:
        from pynvml import (
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetUtilizationRates,
        )

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        utilization = nvmlDeviceGetUtilizationRates(handle)
        gpu_usage.set(utilization.gpu)
    except Exception:
        gpu_usage.set(0)  # Set to 0 if GPU is not available or pynvml is not installed


def find_best_model():
    """Find the best MLflow model based on the lowest MAE."""
    runs = mlflow.search_runs()
    best_run = runs.loc[runs["metrics.mae"].idxmin()]
    return best_run


def load_model(run_id):
    """Load the MLflow model by run ID."""
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from URI: {model_uri}")
    return mlflow.tensorflow.load_model(model_uri)


app = Flask(__name__)


# Endpoint for Prometheus metrics
@app.route("/metrics", methods=["GET"])
def metrics():
    update_hardware_metrics()  # Ensure hardware metrics are updated before exposing them
    return Response(generate_latest(), mimetype="text/plain")


# Endpoint for health checks
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


# Endpoint for predictions with evaluation
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Validate input
    if "input" not in data or "labels" not in data:
        prediction_errors.inc()  # Increment error counter
        return jsonify({"error": "Missing input data or labels"}), 400

    input_data = np.array(data["input"])
    actual_labels = np.array(data["labels"])

    try:
        with prediction_latency.time():  # Measure latency
            predictions = model.predict(input_data).flatten()

        prediction_counter.inc(len(predictions))  # Increment prediction counter

        # Compute metrics
        mae = mean_absolute_error(actual_labels, predictions)
        mse = mean_squared_error(actual_labels, predictions)
        r2 = r2_score(actual_labels, predictions)

        # Update Prometheus metrics
        mae_metric.set(mae)
        mse_metric.set(mse)
        r2_metric.set(r2)

        # Return response
        return jsonify(
            {
                "predictions": predictions.tolist(),
                "mae": mae,
                "mse": mse,
                "r2_score": r2,
            }
        )
    except Exception as e:
        prediction_errors.inc()  # Increment error counter
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    best_run = find_best_model()
    model = load_model(best_run.run_id)

    app.run(host="0.0.0.0", port=5000, debug=True)
