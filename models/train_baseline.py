import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = "data/raw/telco_churn.csv"
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Basic cleaning (minimal only)
# -----------------------------
df = df.drop(columns=["customerID"], errors="ignore")

# Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Handle missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# -----------------------------
# Features / target
# -----------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(),numerical_cols),
    ]
)

# -----------------------------
# Baseline model
# -----------------------------
model = LogisticRegression(max_iter=2000)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

# -----------------------------
# Save metrics
# -----------------------------
with open("models/baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Baseline training complete.")
print(metrics)
