from pathlib import Path
import inspect
import os
import sys
import tempfile
from typing import Any

PREFECT_STATE_DIR = Path(tempfile.gettempdir()) / "churn-mlops-prefect"
PREFECT_STATE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PREFECT_HOME", str(PREFECT_STATE_DIR))
os.environ.setdefault(
    "PREFECT_API_DATABASE_CONNECTION_URL",
    f"sqlite+aiosqlite:///{(PREFECT_STATE_DIR / 'prefect.db').as_posix()}",
)

from pydantic import ConfigDict
import prefect.flows as prefect_flows
import prefect.utilities.callables as prefect_callables
from prefect import flow, task
from prefect.utilities.callables import ParameterSchema, parameter_docstrings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring import alerting, data_drift, prediction_drift


def _patch_prefect_parameter_schema_for_pydantic_v2():
    if not prefect_callables.HAS_PYDANTIC_V2:
        return

    def generate_parameter_schema_compat(signature: inspect.Signature, docstrings: dict[str, str]) -> ParameterSchema:
        model_fields = {}
        aliases = {}
        model_cfg = ConfigDict(arbitrary_types_allowed=True)

        if prefect_callables.has_v1_type_as_param(signature):
            create_schema = prefect_callables.create_v1_schema
            process_params = prefect_callables.process_v1_params

            class ModelConfig:
                arbitrary_types_allowed = True

            model_cfg_to_use: Any = ModelConfig
        else:
            create_schema = prefect_callables.create_v2_schema
            process_params = prefect_callables.process_v2_params
            model_cfg_to_use = model_cfg

        for position, param in enumerate(signature.parameters.values()):
            name, type_, field = process_params(
                param,
                position=position,
                docstrings=docstrings,
                aliases=aliases,
            )
            try:
                create_schema(
                    "CheckParameter",
                    model_cfg=model_cfg_to_use,
                    **{name: (type_, field)},
                )
            except (TypeError, ValueError):
                type_ = Any
            model_fields[name] = (type_, field)

        schema = create_schema("Parameters", model_cfg=model_cfg_to_use, **model_fields)
        return ParameterSchema(**schema)

    def parameter_schema_compat(fn):
        try:
            signature = inspect.signature(fn, eval_str=True)
        except (NameError, TypeError):
            signature = inspect.signature(fn)

        docstrings = parameter_docstrings(inspect.getdoc(fn))
        return generate_parameter_schema_compat(signature, docstrings)

    prefect_callables.generate_parameter_schema = generate_parameter_schema_compat
    prefect_callables.parameter_schema = parameter_schema_compat
    prefect_flows.parameter_schema = parameter_schema_compat


_patch_prefect_parameter_schema_for_pydantic_v2()


@task
def run_data_drift_check():
    """Detect data drift by comparing reference and production distributions.

    Orchestration task that:
    1. Creates production data snapshot
    2. Computes PSI and KS statistics for each feature
    3. Returns drift report

    Returns:
        dict: Data drift report with feature-level metrics.

    Example:
        report = run_data_drift_check()
        # Returns: {"feature1": {"psi": 0.25, "ks_stat": 0.15, ...}, ...}
    """
    data_drift.create_production_snapshot()
    data_drift.detect_data_drift()
    return alerting.load_json(data_drift.REPORT_PATH)


@task
def run_prediction_drift_check():
    """Detect prediction drift by comparing model outputs across distributions.

    Orchestration task that:
    1. Loads model predictions on reference and production data
    2. Computes distribution statistics and mean shift
    3. Returns prediction drift report

    Returns:
        dict: Report with reference/production stats and mean_shift metric.

    Example:
        report = run_prediction_drift_check()
        # Returns: {"reference": {...}, "production": {...}, "mean_shift": 0.05}
    """
    prediction_drift.detect_prediction_drift()
    return alerting.load_json(prediction_drift.REPORT_PATH)


@task
def trigger_alerting_if_needed(data_report, prediction_report):
    """Evaluate drift reports and generate consolidated alerts if needed.

    Checks both data and prediction drift metrics against thresholds.
    If alerts found, triggers full alerting workflow.

    Args:
        data_report: Data drift report dict.
        prediction_report: Prediction drift report dict.

    Returns:
        dict: Contains drift_detected (bool) and total_alerts (int).

    Example:
        result = trigger_alerting_if_needed(data_rep, pred_rep)
        # Returns: {"drift_detected": True, "total_alerts": 2}
    """
    data_alerts = alerting.evaluate_data_drift(data_report)
    prediction_alerts = alerting.evaluate_prediction_drift(prediction_report)
    total_alerts = len(data_alerts) + len(prediction_alerts)

    if total_alerts > 0:
        alerting.run_alerting()

    return {
        "drift_detected": total_alerts > 0,
        "total_alerts": total_alerts,
    }


@flow(name="churn-monitoring-flow")
def churn_monitoring_flow():
    """Main MLOps monitoring workflow for churn prediction system.

    Orchestrates the end-to-end monitoring pipeline:
    1. Data drift detection (PSI, KS tests)
    2. Prediction drift detection (output distribution shifts)
    3. Consolidated alerting (critical/warning levels)

    Returns:
        dict: Pipeline result with drift_detected and total_alerts.

    Example:
        result = churn_monitoring_flow()
        # Returns: {"drift_detected": True, "total_alerts": 2}
    """
    data_report = run_data_drift_check()
    prediction_report = run_prediction_drift_check()
    return trigger_alerting_if_needed(data_report, prediction_report)


if __name__ == "__main__":
    print(churn_monitoring_flow())
