# Churn MLOps System

This project implements a **production-oriented machine learning system** for customer churn prediction, with a strong focus on **reliability, monitoring, and data quality**.

Instead of stopping at model training, the system is designed to reflect the full machine learning lifecycle, including:

- Reusable feature engineering
- Standardized model training and evaluation
- Reference data management
- Statistical data drift detection
- Foundations for monitoring and retraining decisions

The goal of the project is to demonstrate how machine learning models behave **after deployment**, how data changes over time, and how such changes can be detected and acted upon in a controlled, production-ready manner.

## Feature Engineering & Reference Dataset

This project includes a dedicated **feature engineering layer** that transforms raw customer data into model-ready features and produces a **reference feature dataset** representing the training-time data distribution.

### Feature pipeline
A reusable preprocessing pipeline is implemented in `features/build_features.py` with the following transformations:
- Categorical features are encoded using **One-Hot Encoding**
- Numerical features are standardized using **Standard Scaling**
- Feature names are explicitly generated to preserve schema clarity and support monitoring

### Reference dataset
The transformed features are saved as a **reference dataset**:

- `data/reference/reference_features.parquet`

This dataset serves as the **baseline distribution** for:
- Data drift detection
- Comparison with future production data
- Informed model retraining decisions

Separating raw data, feature construction, and reference storage enables reliable monitoring and aligns the system with production-grade MLOps practices.


## Model Training & Evaluation

The project includes standardized scripts for model training and evaluation using the engineered reference features.

### Training
- The model is trained using features from the reference dataset
- The trained model is serialized for reuse and deployment

### Evaluation
- Performance is evaluated on a held-out test split
- Metrics include precision, recall, ROC-AUC, and a confusion matrix
- Evaluation results are persisted for traceability

## Data Drift Detection

The system includes a data drift detection component that compares incoming production features against a reference feature baseline.

### Approach
- The reference feature dataset represents training-time data
- New production features are statistically compared to this baseline
- Drift is quantified using:
  - **Population Stability Index (PSI)**
  - **Kolmogorov–Smirnov (KS) test**

### Output
- Feature-wise drift metrics are generated and persisted
- These metrics provide the foundation for monitoring data quality and triggering retraining decisions

## Prediction Drift Monitoring

Beyond input data drift, the system monitors **prediction drift** by tracking changes in the model’s output probabilities over time.

### Approach
- Prediction probabilities are generated on both reference and production feature sets
- Distribution statistics (mean, spread, range) are compared
- Shifts in prediction behavior are quantified independently of input drift

This enables early detection of silent model degradation, even when input features appear stable.
