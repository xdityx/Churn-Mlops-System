import json
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

REFERENCE_DATA_PATH = "data/reference/reference_features.parquet"
MODEL_PATH = "models/model.pkl"
METRICS_OUTPUT_PATH = "models/evaluation_metrics.json"

def evaluate_model():
    df = pd.read_parquet(REFERENCE_DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation complete.")
    print(metrics)

if __name__ == "__main__":
    evaluate_model()
