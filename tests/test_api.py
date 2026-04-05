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
