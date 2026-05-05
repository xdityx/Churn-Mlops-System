import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

REFERENCE_DATA_PATH = "data/reference/reference_features.parquet"
MODEL_OUTPUT_PATH = "models/model.pkl"


def train_model():
    """Trains a logistic regression model and saves it.

    Loads reference features, splits into train/test sets, trains a logistic
    regression model, and evaluates performance. Saves the trained model to disk
    and returns evaluation metrics.

    Returns:
        dict: Training metrics containing "precision", "recall", "roc_auc",
              and "confusion_matrix".
    """
    df = pd.read_parquet(REFERENCE_DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model trained and saved -> {MODEL_OUTPUT_PATH}")
    print(metrics)

    return metrics


if __name__ == "__main__":
    train_model()
