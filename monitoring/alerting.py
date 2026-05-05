import json
import numpy as np

DATA_DRIFT_REPORT = "monitoring/data_drift_report.json"
PRED_DRIFT_REPORT = "monitoring/prediction_drift_report.json"
ALERT_OUTPUT = "monitoring/alerts.json"


def load_json(path):
    """Loads and returns a JSON file as a dictionary.

    Simple utility function to read JSON from disk.

    Args:
        path: File path to the JSON file.

    Returns:
        dict: Parsed JSON contents as a dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def evaluate_data_drift(drift_report):
    """Evaluates data drift report and generates alerts based on PSI thresholds.

    Parses the drift report and creates critical alerts for PSI >= 0.3 and
    warning alerts for PSI >= 0.2. Each alert includes feature name, PSI value,
    and severity.

    Args:
        drift_report: Dictionary mapping feature names to drift metrics containing
                     "psi" values.

    Returns:
        list: List of alert dictionaries with type, severity, feature, psi, and
              message fields.
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
    """Evaluates prediction drift report and generates alerts based on mean shift.

    Parses the prediction drift report and creates critical alerts for mean shift
    >= 0.10 and warning alerts for mean shift >= 0.05. Each alert documents the
    magnitude of shift and its business impact.

    Args:
        pred_report: Dictionary containing "mean_shift" value calculated from
                    reference vs production predictions.

    Returns:
        list: List of alert dictionaries with type, severity, mean_shift, and
              message fields.
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
    """Orchestrates alerting by evaluating both data and prediction drift reports.

    Loads drift detection reports, evaluates them for alert conditions, combines
    all alerts, and persists the alert summary to a JSON file.

    Returns:
        None
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
