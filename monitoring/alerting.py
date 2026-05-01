"""Alert generation from drift detection reports."""
import json
import numpy as np

DATA_DRIFT_REPORT = "monitoring/data_drift_report.json"
PRED_DRIFT_REPORT = "monitoring/prediction_drift_report.json"
ALERT_OUTPUT = "monitoring/alerts.json"


def load_json(path):
    """Load JSON report from disk.

    Args:
        path: File path to JSON report.

    Returns:
        dict: Parsed JSON content.

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If file is not valid JSON.

    Example:
        report = load_json("monitoring/data_drift_report.json")
    """
    with open(path, "r") as f:
        return json.load(f)


def evaluate_data_drift(drift_report):
    """Generate alerts based on feature-level data drift metrics.

    Evaluates Population Stability Index (PSI) for each feature:
    - PSI >= 0.3: critical severity, retraining recommended
    - PSI >= 0.2: warning severity

    Args:
        drift_report: Dict mapping feature names to {psi, ks_stat, ks_pvalue}.

    Returns:
        list: Alert dicts with type, severity, feature, psi, message fields.

    Example:
        alerts = evaluate_data_drift({"feature1": {"psi": 0.35, ...}})
        # Returns: [{"type": "data_drift", "severity": "critical", ...}]
    """
    alerts = []

    for feature, metrics in drift_report.items():
        psi = metrics["psi"]

        if psi >= 0.3:
            alerts.append({
                "type": "data_drift",
                "severity": "critical",
                "feature": feature,
                "psi": psi,
                "message": "Severe data drift detected. Retraining recommended."
            })
        elif psi >= 0.2:
            alerts.append({
                "type": "data_drift",
                "severity": "warning",
                "feature": feature,
                "psi": psi,
                "message": "Moderate data drift detected."
            })

    return alerts


def evaluate_prediction_drift(pred_report):
    """Generate alerts based on model prediction distribution shift.

    Evaluates mean_shift (difference in mean prediction probability):
    - mean_shift >= 0.10: critical severity, business risk elevated
    - mean_shift >= 0.05: warning severity

    Args:
        pred_report: Dict with reference, production stats and mean_shift.

    Returns:
        list: Alert dicts with type, severity, mean_shift, message fields.

    Example:
        alerts = evaluate_prediction_drift({"mean_shift": 0.12, ...})
        # Returns: [{"type": "prediction_drift", "severity": "critical", ...}]
    """
    alerts = []

    mean_shift = abs(pred_report.get("mean_shift", 0.0))

    if mean_shift >= 0.10:
        alerts.append({
            "type": "prediction_drift",
            "severity": "critical",
            "mean_shift": mean_shift,
            "message": "Severe prediction drift detected. Business risk elevated."
        })
    elif mean_shift >= 0.05:
        alerts.append({
            "type": "prediction_drift",
            "severity": "warning",
            "mean_shift": mean_shift,
            "message": "Moderate prediction behavior shift detected."
        })

    return alerts


def run_alerting():
    """Load drift reports, evaluate for alerts, save consolidated alert state.

    Combines data drift and prediction drift alerts into single alert output.

    Returns:
        None. Saves comprehensive alert state to ALERT_OUTPUT.

    Raises:
        FileNotFoundError: If drift report paths do not exist.

    Example:
        run_alerting()
        # Outputs: Alerting complete. Saves alerts.json with all alerts.
    """
    data_drift = load_json(DATA_DRIFT_REPORT)
    pred_drift = load_json(PRED_DRIFT_REPORT)

    alerts = []
    alerts.extend(evaluate_data_drift(data_drift))
    alerts.extend(evaluate_prediction_drift(pred_drift))

    output = {
        "total_alerts": len(alerts),
        "alerts": alerts
    }

    with open(ALERT_OUTPUT, "w") as f:
        json.dump(output, f, indent=4)

    print("Alerting complete.")
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    run_alerting()
