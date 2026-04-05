def test_model_trains_and_returns_valid_metrics(trained_artifacts):
    metrics = trained_artifacts["metrics"]

    assert trained_artifacts["model_path"].exists()
    assert set(metrics) == {"precision", "recall", "roc_auc", "confusion_matrix"}
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert all(len(row) == 2 for row in metrics["confusion_matrix"])
