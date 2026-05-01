import importlib
import sys

import joblib
import pandas as pd
from fastapi.testclient import TestClient


def test_predict_endpoint_returns_200(reference_artifacts, trained_artifacts, monkeypatch):
    reference_df = reference_artifacts["reference_df"]
    model = trained_artifacts["model"]

    monkeypatch.setattr(joblib, "load", lambda _: model)
    monkeypatch.setattr(pd, "read_parquet", lambda _: reference_df)

    sys.modules.pop("api.app", None)
    api_module = importlib.import_module("api.app")
    client = TestClient(api_module.app)

    row = reference_df.drop(columns=["Churn"]).iloc[0]
    payload = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.items()
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {0, 1}


def test_health_endpoint_returns_ok(reference_artifacts, trained_artifacts, monkeypatch):
    reference_df = reference_artifacts["reference_df"]
    model = trained_artifacts["model"]

    monkeypatch.setattr(joblib, "load", lambda _: model)
    monkeypatch.setattr(pd, "read_parquet", lambda _: reference_df)

    sys.modules.pop("api.app", None)
    api_module = importlib.import_module("api.app")
    client = TestClient(api_module.app)

    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "Churn MLOps API is running" in body["message"]


def test_predict_with_invalid_input_types(reference_artifacts, trained_artifacts, monkeypatch):
    reference_df = reference_artifacts["reference_df"]
    model = trained_artifacts["model"]

    monkeypatch.setattr(joblib, "load", lambda _: model)
    monkeypatch.setattr(pd, "read_parquet", lambda _: reference_df)

    sys.modules.pop("api.app", None)
    api_module = importlib.import_module("api.app")
    client = TestClient(api_module.app)

    row = reference_df.drop(columns=["Churn"]).iloc[0]
    partial_payload = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in list(row.items())[:2]
    }

    response = client.post("/predict", json=partial_payload)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body


def test_predict_with_missing_fields(reference_artifacts, trained_artifacts, monkeypatch):
    reference_df = reference_artifacts["reference_df"]
    model = trained_artifacts["model"]

    monkeypatch.setattr(joblib, "load", lambda _: model)
    monkeypatch.setattr(pd, "read_parquet", lambda _: reference_df)

    sys.modules.pop("api.app", None)
    api_module = importlib.import_module("api.app")
    client = TestClient(api_module.app)

    row = reference_df.drop(columns=["Churn"]).iloc[0]
    partial_payload = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in list(row.items())[:3]
    }

    response = client.post("/predict", json=partial_payload)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert "churn_prediction" in body


def test_alerts_endpoint_schema_validation(reference_artifacts, trained_artifacts, monkeypatch, tmp_path):
    reference_df = reference_artifacts["reference_df"]
    model = trained_artifacts["model"]
    alerts_file = tmp_path / "alerts.json"

    alerts_file.write_text('{"total_alerts": 2, "alerts": [{"type": "data_drift", "severity": "warning"}]}')

    monkeypatch.setattr(joblib, "load", lambda _: model)
    monkeypatch.setattr(pd, "read_parquet", lambda _: reference_df)
    monkeypatch.setenv("ALERTS_PATH", str(alerts_file))

    sys.modules.pop("api.app", None)
    import api.app as api_module
    monkeypatch.setattr(api_module, "ALERTS_PATH", str(alerts_file))

    client = TestClient(api_module.app)
    response = client.get("/alerts")

    assert response.status_code == 200
    body = response.json()
    assert "total_alerts" in body
    assert "alerts" in body
    assert isinstance(body["alerts"], list)
