import json
import joblib
import numpy as np
import pandas as pd

REFERENCE_DATA_PATH = "data/reference/reference_features.parquet"
PRODUCTION_DATA_PATH = "data/production/production_features.parquet"
MODEL_PATH = "models/model.pkl"
REPORT_PATH = "monitoring/prediction_drift_report.json"


def load_predictions(data_path):
    """Loads data and returns model predictions (churn probabilities).

    Reads parquet data, removes the target column, loads the trained model,
    and generates churn probability predictions for all samples.

    Args:
        data_path: File path to the parquet data file.

    Returns:
        numpy.ndarray: Churn probability predictions for each sample (values 0-1).
    """
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["Churn"], errors="ignore")

    model = joblib.load(MODEL_PATH)
    preds = model.predict_proba(X)[:, 1]

    return preds


def calculate_distribution_stats(preds):
    """Calculates summary statistics of prediction distribution.

    Computes mean, standard deviation, minimum, and maximum values for a set of
    predictions to characterize the distribution shape.

    Args:
        preds: Array of prediction values.

    Returns:
        dict: Dictionary with "mean", "std", "min", and "max" keys.
    """
    return {
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "min": float(np.min(preds)),
        "max": float(np.max(preds)),
    }


def detect_prediction_drift():
    """Detects prediction drift by comparing reference and production predictions.

    Generates predictions on both reference and production datasets, computes
    distribution statistics for each, and calculates mean shift. Persists the
    comprehensive drift report to disk.

    Returns:
        None
    """
    ref_preds = load_predictions(REFERENCE_DATA_PATH)
    prod_preds = load_predictions(PRODUCTION_DATA_PATH)

    report = {
        "reference": calculate_distribution_stats(ref_preds),
        "production": calculate_distribution_stats(prod_preds),
        "mean_shift": float(np.mean(prod_preds) - np.mean(ref_preds)),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Prediction drift report saved -> {REPORT_PATH}")
    print(report)


if __name__ == "__main__":
    detect_prediction_drift()
