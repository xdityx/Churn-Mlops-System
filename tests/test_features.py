import pandas as pd
import numpy as np
import pytest


def test_build_features_runs_and_output_shape_is_valid(reference_artifacts):
    reference_df = reference_artifacts["reference_df"]

    assert reference_artifacts["reference_path"].exists()
    assert reference_df.shape[0] == 50
    assert "Churn" in reference_df.columns
    assert reference_df.drop(columns=["Churn"]).shape[1] >= 5


def test_build_features_preserves_churn_label(tmp_path, monkeypatch):
    import features.build_features as build_features_module

    churn_df_path = tmp_path / "churn_check.csv"
    df = pd.DataFrame({
        "customerID": [f"CUST-{i}" for i in range(10)],
        "gender": ["Male", "Female"] * 5,
        "tenure": [12, 24, 36, 48, 60] * 2,
        "MonthlyCharges": [50, 65, 75, 85, 95] * 2,
        "TotalCharges": ["600", "1560", "2700", "4080", "5700"] * 2,
        "Churn": ["Yes", "No"] * 5,
    })
    df.to_csv(churn_df_path, index=False)

    reference_path = tmp_path / "reference_churn.parquet"
    monkeypatch.setattr(build_features_module, "RAW_DATA_PATH", str(churn_df_path))
    monkeypatch.setattr(build_features_module, "REFERENCE_OUTPUT_PATH", str(reference_path))

    build_features_module.build_reference_features()
    result_df = pd.read_parquet(reference_path)

    assert "Churn" in result_df.columns
    assert result_df["Churn"].unique().tolist() == [1, 0] or result_df["Churn"].unique().tolist() == [0, 1]


def test_build_features_removes_nan_values(reference_artifacts, tmp_path, monkeypatch):
    import features.build_features as build_features_module

    nan_raw_path = tmp_path / "nan_churn.csv"
    nan_df = pd.DataFrame({
        "customerID": [f"CUST-{i}" for i in range(15)],
        "gender": ["Male", "Female", None] * 5,
        "tenure": [12, 24, None, 48, 60] * 3,
        "TotalCharges": ["1200", "2400", "invalid", "4800", "6000"] * 3,
        "Churn": ["Yes", "No"] * 7 + ["No"],
    })
    nan_df.to_csv(nan_raw_path, index=False)

    reference_path = tmp_path / "reference_features_nan.parquet"
    monkeypatch.setattr(build_features_module, "RAW_DATA_PATH", str(nan_raw_path))
    monkeypatch.setattr(build_features_module, "REFERENCE_OUTPUT_PATH", str(reference_path))

    build_features_module.build_reference_features()
    result_df = pd.read_parquet(reference_path)

    assert not result_df.isnull().any().any()
    assert result_df.shape[0] > 0


def test_build_features_output_dtypes_are_numeric(reference_artifacts):
    reference_df = reference_artifacts["reference_df"]

    numeric_cols = reference_df.drop(columns=["Churn"]).columns
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(reference_df[col]), f"Column {col} should be numeric"

    assert pd.api.types.is_numeric_dtype(reference_df["Churn"])
