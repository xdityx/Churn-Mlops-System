"""Prediction drift detection comparing model outputs across distributions."""
import json
import joblib
import numpy as np
import pandas as pd

REFERENCE_DATA_PATH = "data/reference/reference_features.parquet"
PRODUCTION_DATA_PATH = "data/production/production_features.parquet"
MODEL_PATH = "models/model.pkl"
REPORT_PATH = "monitoring/prediction_drift_report.json"


def load_predictions(data_path):
    """Load data and generate predictions from trained model.

    Args:
        data_path: Path to features parquet file.

    Returns:
        np.ndarray: Predicted churn probabilities (shape: (n_samples,)).

    Raises:
        FileNotFoundError: If data_path or MODEL_PATH do not exist.

    Example:
        preds = load_predictions("data/production/production_features.parquet")
        # Returns array of probabilities
    """
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["Churn"], errors="ignore")

    model = joblib.load(MODEL_PATH)
    preds = model.predict_proba(X)[:, 1]

    return preds


def calculate_distribution_stats(preds):
    """Compute summary statistics of prediction distribution.

    Args:
        preds: Array of predicted probabilities.

    Returns:
        dict: Contains mean, std, min, max as floats.

    Example:
        stats = calculate_distribution_stats(preds)
        # Returns: {"mean": 0.45, "std": 0.2, "min": 0.01, "max": 0.99}
    """
    return {
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "min": float(np.min(preds)),
        "max": float(np.max(preds)),
    }


def detect_prediction_drift():
    """Compare model prediction distributions between reference and production.

    Computes distribution statistics for both reference (training) and production
    (recent) predictions. Reports mean_shift as key indicator of model performance drift.

    Returns:
        None. Saves report to REPORT_PATH with reference stats, production stats,
              and mean_shift metric.

    Raises:
        FileNotFoundError: If data or model paths do not exist.

    Example:
        detect_prediction_drift()
        # Outputs: Prediction drift report saved -> monitoring/prediction_drift_report.json
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
