"""Churn MLOps API for model inference and monitoring exposure."""
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
    """Health check endpoint.

    Returns:
        dict: Status and message indicating API is running.

    Example:
        GET / -> {"status": "ok", "message": "Churn MLOps API is running"}
    """
    return {"status": "ok", "message": "Churn MLOps API is running"}


@app.post("/predict")
def predict(payload: dict):
    """Generate churn prediction from customer features.

    Args:
        payload: JSON object with feature_name: value pairs matching reference schema.

    Returns:
        dict: Contains churn_probability (float 0-1) and churn_prediction (0 or 1).

    Raises:
        ValueError: If required features are missing or invalid.

    Example:
        POST /predict with {"tenure": 12, "MonthlyCharges": 65.5, ...}
        -> {"churn_probability": 0.72, "churn_prediction": 1}
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
    """Retrieve latest monitoring and alerting state.

    Returns:
        dict: Contains total_alerts count and list of alert objects.
              Each alert has type, severity, and message fields.

    Example:
        GET /alerts -> {"total_alerts": 2, "alerts": [...]}
    """
    try:
        with open(ALERTS_PATH, "r") as f:
            alerts = json.load(f)
    except FileNotFoundError:
        alerts = {"total_alerts": 0, "alerts": []}

    return alerts
