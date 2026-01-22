## Model Training & Evaluation

The project includes standardized scripts for model training and evaluation using the engineered reference features.

### Training
- The model is trained using features from the reference dataset
- The trained model is serialized for reuse and deployment

### Evaluation
- Performance is evaluated on a held-out test split
- Metrics include precision, recall, ROC-AUC, and a confusion matrix
- Evaluation results are persisted for traceability
