import json

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
