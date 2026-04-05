def test_build_features_runs_and_output_shape_is_valid(reference_artifacts):
    reference_df = reference_artifacts["reference_df"]

    assert reference_artifacts["reference_path"].exists()
    assert reference_df.shape[0] == 50
    assert "Churn" in reference_df.columns
    assert reference_df.drop(columns=["Churn"]).shape[1] >= 5
