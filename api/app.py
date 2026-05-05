from fastapi import FastAPI
import joblib
import json
import pandas as pd

MODEL_PATH = "models/model.pkl"
ALERTS_PATH = "monitoring/alerts.json"
REFERENCE_SCHEMA_PATH = "data/reference/reference_features.parquet"

app = FastAPI(
    title="Churn MLOps System API",
    description="Inference and monitoring exposure for churn prediction system",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)

# Load schema for feature order
_reference_df = pd.read_parquet(REFERENCE_SCHEMA_PATH)
FEATURE_COLUMNS = [c for c in _reference_df.columns if c != "Churn"]


@app.get("/")
def health():
    """Returns API health status.

    Indicates whether the Churn MLOps API service is operational and ready to
    accept requests.

    Returns:
        dict: Status dictionary with "status" and "message" fields.
    """
    return {"status": "ok", "message": "Churn MLOps API is running"}


@app.post("/predict")
def predict(payload: dict):
    """Predicts churn probability for a customer.

    Accepts customer features and returns the probability of churn along with
    a binary prediction. Missing features are filled with zero values.

    Args:
        payload: Dictionary with feature names as keys and feature values.

    Returns:
        dict: Dictionary containing "churn_probability" (float 0-1) and
              "churn_prediction" (int 0 or 1).
    """
    df = pd.DataFrame([payload])

    # enforce feature order and missing handling
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    return {
        "churn_probability": prob,
        "churn_prediction": int(prob >= 0.5),
    }


@app.get("/alerts")
def get_alerts():
    """Returns the latest alerting state.

    Reads the alerts JSON file to return all currently active alerts from data
    and prediction drift detection. Returns an empty alert list if no alerts exist.

    Returns:
        dict: Dictionary with "total_alerts" count and "alerts" list of
              alert objects.
    """
    try:
        with open(ALERTS_PATH, "r") as f:
            alerts = json.load(f)
    except FileNotFoundError:
        alerts = {"total_alerts": 0, "alerts": []}

    return alerts
