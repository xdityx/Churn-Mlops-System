import joblib
import numpy as np
import pandas as pd
import pytest

import features.build_features as build_features_module
import models.train as train_module


def make_synthetic_raw_df(rows=50):
    rng = np.random.default_rng(42)
    churn = np.array([0, 1] * (rows // 2), dtype=int)
    monthly_charges = np.where(
        churn == 1,
        rng.normal(88, 6, size=rows),
        rng.normal(46, 5, size=rows),
    ).round(2)
    tenure = np.where(
        churn == 1,
        rng.integers(1, 18, size=rows),
        rng.integers(24, 72, size=rows),
    )
    total_charges = (monthly_charges * tenure).round(2)

    return pd.DataFrame(
        {
            "customerID": [f"CUST-{i:03d}" for i in range(rows)],
            "gender": np.where(np.arange(rows) % 2 == 0, "Female", "Male"),
            "SeniorCitizen": (np.arange(rows) % 5 == 0).astype(int),
            "Partner": np.where(churn == 1, "No", "Yes"),
            "InternetService": np.where(churn == 1, "Fiber optic", "DSL"),
            "TechSupport": np.where(churn == 1, "No", "Yes"),
            "Contract": np.where(churn == 1, "Month-to-month", "Two year"),
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges.astype(str),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


@pytest.fixture
def reference_artifacts(tmp_path, monkeypatch):
    raw_path = tmp_path / "telco_churn.csv"
    reference_path = tmp_path / "reference_features.parquet"
    raw_df = make_synthetic_raw_df()
    raw_df.to_csv(raw_path, index=False)

    monkeypatch.setattr(build_features_module, "RAW_DATA_PATH", str(raw_path))
    monkeypatch.setattr(build_features_module, "REFERENCE_OUTPUT_PATH", str(reference_path))

    build_features_module.build_reference_features()
    reference_df = pd.read_parquet(reference_path)

    return {
        "raw_df": raw_df,
        "raw_path": raw_path,
        "reference_df": reference_df,
        "reference_path": reference_path,
    }


@pytest.fixture
def trained_artifacts(tmp_path, monkeypatch, reference_artifacts):
    model_path = tmp_path / "model.pkl"

    monkeypatch.setattr(train_module, "REFERENCE_DATA_PATH", str(reference_artifacts["reference_path"]))
    monkeypatch.setattr(train_module, "MODEL_OUTPUT_PATH", str(model_path))

    metrics = train_module.train_model()
    model = joblib.load(model_path)

    return {
        "metrics": metrics,
        "model": model,
        "model_path": model_path,
    }
