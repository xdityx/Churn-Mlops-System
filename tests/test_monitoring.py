import json
import numpy as np
import pandas as pd

import monitoring.alerting as alerting_module
import monitoring.data_drift as data_drift_module
import monitoring.prediction_drift as prediction_drift_module


def test_drift_detection_runs_without_errors(tmp_path, monkeypatch, reference_artifacts, trained_artifacts):
    production_path = tmp_path / "production_features.parquet"
    data_report_path = tmp_path / "data_drift_report.json"
    prediction_report_path = tmp_path / "prediction_drift_report.json"
    alerts_path = tmp_path / "alerts.json"

    monkeypatch.setattr(data_drift_module, "REFERENCE_PATH", str(reference_artifacts["reference_path"]))
    monkeypatch.setattr(data_drift_module, "PRODUCTION_PATH", str(production_path))
    monkeypatch.setattr(data_drift_module, "REPORT_PATH", str(data_report_path))

    monkeypatch.setattr(prediction_drift_module, "REFERENCE_DATA_PATH", str(reference_artifacts["reference_path"]))
    monkeypatch.setattr(prediction_drift_module, "PRODUCTION_DATA_PATH", str(production_path))
    monkeypatch.setattr(prediction_drift_module, "MODEL_PATH", str(trained_artifacts["model_path"]))
    monkeypatch.setattr(prediction_drift_module, "REPORT_PATH", str(prediction_report_path))

    monkeypatch.setattr(alerting_module, "DATA_DRIFT_REPORT", str(data_report_path))
    monkeypatch.setattr(alerting_module, "PRED_DRIFT_REPORT", str(prediction_report_path))
    monkeypatch.setattr(alerting_module, "ALERT_OUTPUT", str(alerts_path))

    data_drift_module.create_production_snapshot()
    data_drift_module.detect_data_drift()
    prediction_drift_module.detect_prediction_drift()
    alerting_module.run_alerting()

    assert production_path.exists()
    assert data_report_path.exists()
    assert prediction_report_path.exists()
    assert alerts_path.exists()

    alerts = json.loads(alerts_path.read_text())
    assert "total_alerts" in alerts
    assert "alerts" in alerts


def test_calculate_psi_edge_cases():
    expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    actual = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    psi = data_drift_module.calculate_psi(expected, actual)
    assert psi == 0.0 or psi < 0.01


def test_calculate_psi_with_distributions():
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0.2, 1, 1000)

    psi = data_drift_module.calculate_psi(expected, actual)
    assert isinstance(psi, (int, float))
    assert psi >= 0


def test_evaluate_data_drift_alert_thresholds():
    drift_report = {
        "feature1": {"psi": 0.35, "ks_stat": 0.2, "ks_pvalue": 0.01},
        "feature2": {"psi": 0.22, "ks_stat": 0.15, "ks_pvalue": 0.05},
        "feature3": {"psi": 0.05, "ks_stat": 0.1, "ks_pvalue": 0.5},
    }

    alerts = alerting_module.evaluate_data_drift(drift_report)

    critical_alerts = [a for a in alerts if a["severity"] == "critical"]
    warning_alerts = [a for a in alerts if a["severity"] == "warning"]

    assert len(critical_alerts) >= 1
    assert len(warning_alerts) >= 1


def test_evaluate_prediction_drift_alert_thresholds():
    pred_report = {
        "reference": {"mean": 0.45, "std": 0.2, "min": 0.01, "max": 0.99},
        "production": {"mean": 0.56, "std": 0.22, "min": 0.02, "max": 0.98},
        "mean_shift": 0.11,
    }

    alerts = alerting_module.evaluate_prediction_drift(pred_report)

    assert len(alerts) >= 1
    assert any(a["severity"] == "critical" for a in alerts)


def test_alerts_report_schema_validity(tmp_path, monkeypatch, reference_artifacts, trained_artifacts):
    production_path = tmp_path / "production_features.parquet"
    data_report_path = tmp_path / "data_drift_report.json"
    prediction_report_path = tmp_path / "prediction_drift_report.json"
    alerts_path = tmp_path / "alerts.json"

    monkeypatch.setattr(data_drift_module, "REFERENCE_PATH", str(reference_artifacts["reference_path"]))
    monkeypatch.setattr(data_drift_module, "PRODUCTION_PATH", str(production_path))
    monkeypatch.setattr(data_drift_module, "REPORT_PATH", str(data_report_path))

    monkeypatch.setattr(prediction_drift_module, "REFERENCE_DATA_PATH", str(reference_artifacts["reference_path"]))
    monkeypatch.setattr(prediction_drift_module, "PRODUCTION_DATA_PATH", str(production_path))
    monkeypatch.setattr(prediction_drift_module, "MODEL_PATH", str(trained_artifacts["model_path"]))
    monkeypatch.setattr(prediction_drift_module, "REPORT_PATH", str(prediction_report_path))

    monkeypatch.setattr(alerting_module, "DATA_DRIFT_REPORT", str(data_report_path))
    monkeypatch.setattr(alerting_module, "PRED_DRIFT_REPORT", str(prediction_report_path))
    monkeypatch.setattr(alerting_module, "ALERT_OUTPUT", str(alerts_path))

    data_drift_module.create_production_snapshot()
    data_drift_module.detect_data_drift()
    prediction_drift_module.detect_prediction_drift()
    alerting_module.run_alerting()

    alerts = json.loads(alerts_path.read_text())

    assert isinstance(alerts, dict)
    assert "total_alerts" in alerts
    assert isinstance(alerts["total_alerts"], int)
    assert "alerts" in alerts
    assert isinstance(alerts["alerts"], list)

    for alert in alerts["alerts"]:
        assert "type" in alert
        assert "severity" in alert
        assert alert["type"] in ["data_drift", "prediction_drift"]
        assert alert["severity"] in ["critical", "warning"]
        assert "message" in alert
