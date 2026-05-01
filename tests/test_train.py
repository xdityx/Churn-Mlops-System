import joblib
import numpy as np
import pandas as pd
import pytest

import models.train as train_module


def test_model_trains_and_returns_valid_metrics(trained_artifacts):
    metrics = trained_artifacts["metrics"]

    assert trained_artifacts["model_path"].exists()
    assert set(metrics) == {"precision", "recall", "roc_auc", "confusion_matrix"}
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert all(len(row) == 2 for row in metrics["confusion_matrix"])


def test_model_training_reproducibility(tmp_path, monkeypatch, reference_artifacts):
    model_path_1 = tmp_path / "model_1.pkl"
    model_path_2 = tmp_path / "model_2.pkl"

    monkeypatch.setattr(train_module, "REFERENCE_DATA_PATH", str(reference_artifacts["reference_path"]))
    monkeypatch.setattr(train_module, "MODEL_OUTPUT_PATH", str(model_path_1))
    metrics_1 = train_module.train_model()

    monkeypatch.setattr(train_module, "MODEL_OUTPUT_PATH", str(model_path_2))
    metrics_2 = train_module.train_model()

    assert metrics_1["precision"] == metrics_2["precision"]
    assert metrics_1["recall"] == metrics_2["recall"]
    assert metrics_1["roc_auc"] == metrics_2["roc_auc"]


def test_model_metrics_are_valid_ranges(trained_artifacts):
    metrics = trained_artifacts["metrics"]

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    cm = metrics["confusion_matrix"]
    assert all(val >= 0 for row in cm for val in row)
    assert sum(sum(row) for row in cm) > 0


def test_trained_model_can_be_loaded_from_pickle(trained_artifacts):
    model_path = trained_artifacts["model_path"]
    model = joblib.load(model_path)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert model is not None


def test_model_file_is_saved_successfully(trained_artifacts):
    model_path = trained_artifacts["model_path"]

    assert model_path.exists()
    assert model_path.stat().st_size > 0

    model = joblib.load(model_path)
    assert model is not None
