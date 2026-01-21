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
