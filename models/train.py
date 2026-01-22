import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

REFERENCE_DATA_PATH = "data/reference/reference_features.parquet"
MODEL_OUTPUT_PATH = "models/model.pkl"

def train_model():
    df = pd.read_parquet(REFERENCE_DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model trained and saved → {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train_model()
