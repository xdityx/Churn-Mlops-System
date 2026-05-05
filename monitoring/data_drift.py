import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_PATH = "data/reference/reference_features.parquet"
PRODUCTION_PATH = "data/production/production_features.parquet"
REPORT_PATH = "monitoring/data_drift_report.json"


def calculate_psi(expected, actual, buckets=10):
    """Calculates Population Stability Index (PSI) between two distributions.

    Computes PSI metric to quantify distributional shift. PSI > 0.1 indicates
    significant change, PSI > 0.25 indicates major change. Uses histogram-based
    binning with log-ratio of percentages.

    Args:
        expected: Reference distribution array.
        actual: Actual/production distribution array.
        buckets: Number of histogram bins for comparison (default 10).

    Returns:
        float: PSI value measuring distribution divergence.
    """
    expected_percents, _ = np.histogram(expected, bins=buckets)
    actual_percents, _ = np.histogram(actual, bins=buckets)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi = np.sum(
        (expected_percents - actual_percents)
        * np.log((expected_percents + 1e-6) / (actual_percents + 1e-6))
    )
    return psi


def create_production_snapshot():
    """Creates a synthetic production dataset with simulated data drift.

    Loads reference data, samples 80% of it, applies a normal multiplier (mean
    1.05, std 0.02) to numeric features to simulate production shift. Saves
    the drifted snapshot for drift detection benchmarking.

    Returns:
        None
    """
    ref = pd.read_parquet(REFERENCE_PATH)

    prod = ref.sample(frac=0.8, random_state=42).copy()

    numeric_cols = prod.select_dtypes(include="number").columns.tolist()
    numeric_cols.remove("Churn")

    for col in numeric_cols:
        prod[col] = prod[col] * np.random.normal(1.05, 0.02)

    prod.to_parquet(PRODUCTION_PATH, index=False)
    print(f"Production snapshot saved -> {PRODUCTION_PATH}")


def detect_data_drift():
    """Detects data drift by comparing reference and production feature distributions.

    Compares each feature between reference and production data using PSI and
    Kolmogorov-Smirnov test. Generates a comprehensive drift report with both
    metrics and persists it to disk.

    Returns:
        None
    """
    ref = pd.read_parquet(REFERENCE_PATH)
    prod = pd.read_parquet(PRODUCTION_PATH)

    drift_report = {}
    feature_cols = [c for c in ref.columns if c != "Churn"]

    for col in feature_cols:
        ref_col = ref[col]
        prod_col = prod[col]

        psi = calculate_psi(ref_col, prod_col)
        ks_stat, ks_pvalue = ks_2samp(ref_col, prod_col)

        drift_report[col] = {
            "psi": float(psi),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
        }

    with open(REPORT_PATH, "w") as f:
        json.dump(drift_report, f, indent=4)

    print(f"Data drift report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    create_production_snapshot()
    detect_data_drift()
