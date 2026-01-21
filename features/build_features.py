import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = "data/raw/telco_churn.csv"
REFERENCE_OUTPUT_PATH = "data/reference/reference_features.parquet"


def build_reference_features():
    df = pd.read_csv(RAW_DATA_PATH)

    # -----------------------------
    # Minimal cleaning (locked)
    # -----------------------------
    df = df.drop(columns=["customerID"], errors="ignore")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # -----------------------------
    # Feature pipeline
    # -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    feature_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(categorical_cols)
        .tolist()
        + numerical_cols
    )

    feature_df = pd.DataFrame(X_processed, columns=feature_names)
    feature_df["Churn"] = y.reset_index(drop=True)

    feature_df.to_parquet(REFERENCE_OUTPUT_PATH, index=False)
    print(f"Reference features saved → {REFERENCE_OUTPUT_PATH}")


if __name__ == "__main__":
    build_reference_features()
