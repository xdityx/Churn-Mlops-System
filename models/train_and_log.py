from pathlib import Path
import sys

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.build_features import build_reference_features
from models.evaluate import evaluate_model
from models.train import train_model

MODEL_TYPE = "LogisticRegression"
TEST_SIZE = 0.2
RANDOM_SEED = 42
TRACKING_URI = (PROJECT_ROOT / "mlruns").resolve().as_uri()
EXPERIMENT_NAME = "churn-mlops"


def train_and_log():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    build_reference_features()

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": MODEL_TYPE,
                "test_size": TEST_SIZE,
                "random_seed": RANDOM_SEED,
            }
        )

        train_model()
        metrics = evaluate_model()

        mlflow.log_metrics(
            {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    return metrics


if __name__ == "__main__":
    train_and_log()
