import json
import joblib
import numpy as np
import pandas as pd

REFERENCE_DATA_PATH = "data/reference/reference_features.parquet"
PRODUCTION_DATA_PATH = "data/production/production_features.parquet"
MODEL_PATH = "models/model.pkl"
REPORT_PATH = "monitoring/prediction_drift_report.json"


def load_predictions(data_path):
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["Churn"], errors="ignore")

    model = joblib.load(MODEL_PATH)
    preds = model.predict_proba(X)[:, 1]

    return preds


def calculate_distribution_stats(preds):
    return {
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "min": float(np.min(preds)),
        "max": float(np.max(preds)),
    }


def detect_prediction_drift():
    ref_preds = load_predictions(REFERENCE_DATA_PATH)
    prod_preds = load_predictions(PRODUCTION_DATA_PATH)

    report = {
        "reference": calculate_distribution_stats(ref_preds),
        "production": calculate_distribution_stats(prod_preds),
        "mean_shift": float(np.mean(prod_preds) - np.mean(ref_preds)),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Prediction drift report saved → {REPORT_PATH}")
    print(report)


if __name__ == "__main__":
    detect_prediction_drift()
