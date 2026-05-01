"""Data drift detection using PSI and Kolmogorov-Smirnov tests."""
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_PATH = "data/reference/reference_features.parquet"
PRODUCTION_PATH = "data/production/production_features.parquet"
REPORT_PATH = "monitoring/data_drift_report.json"


def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index between two distributions.

    PSI measures distributional shift. Values >0.3 indicate significant drift.

    Args:
        expected: Reference distribution array.
        actual: Production distribution array.
        buckets: Number of histogram bins (default: 10).

    Returns:
        float: PSI value. Interpretation: <0.1 (no drift), 0.1-0.2 (small),
               0.2-0.3 (moderate), >0.3 (significant).

    Example:
        psi = calculate_psi(ref_data, prod_data)  # Returns: 0.25
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
    """Create simulated production data by sampling and perturbing reference data.

    Samples 80% of reference data and applies random Gaussian noise to numeric
    features (mean=1.05, std=0.02) to simulate realistic production drift.

    Returns:
        None. Saves production snapshot to PRODUCTION_PATH.

    Raises:
        FileNotFoundError: If REFERENCE_PATH does not exist.

    Example:
        create_production_snapshot()
        # Outputs: Production snapshot saved -> data/production/production_features.parquet
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
    """Compare reference and production distributions, compute PSI and KS statistics.

    Calculates Population Stability Index and Kolmogorov-Smirnov test statistics
    for each feature to detect distributional shifts between training and production data.

    Returns:
        None. Saves drift report (dict of feature -> metrics) to REPORT_PATH.

    Raises:
        FileNotFoundError: If REFERENCE_PATH or PRODUCTION_PATH do not exist.

    Example:
        detect_data_drift()
        # Outputs: Data drift report saved -> monitoring/data_drift_report.json
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
