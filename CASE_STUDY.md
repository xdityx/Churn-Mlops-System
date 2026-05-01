# Churn MLOps System - Case Study

## The Problem

Machine learning models rarely fail in obvious ways. A churn model can validate well at launch and still become unreliable a few months later because customer behavior changes. Seasonality, pricing shifts, product launches, macroeconomic pressure, and acquisition mix all change feature distributions over time.

That is especially dangerous in churn prediction because the model still returns scores, even when the assumptions behind those scores are no longer true. A model trained on 2025 behavior may underperform on 2026 behavior, especially during events like a recession or a major product change. The result is poor targeting, wasted retention spend, and late strategic surprises.

## Why This Matters

MLOps matters because model degradation is usually silent. Unlike a software bug, it does not crash the system; it quietly makes decisions worse. For subscription businesses, even a small increase in churn can create meaningful revenue loss, so catching drift in days instead of weeks has real financial value.

Many teams still retrain annually or only when someone notices KPI changes. That is reactive and expensive. Strong notebooks and good offline metrics are not enough. Without monitoring, alerting, and repeatable workflows, the model is not production-ready.

## What I Built

This project is a production-oriented churn prediction system designed to show what real ML engineering looks like beyond model training. It starts with raw telco churn data, transforms it into a repeatable reference feature set, trains a churn model, exposes predictions through an API, and monitors for drift after deployment.

The data layer separates raw inputs from engineered features. A reusable feature pipeline standardizes numerical variables, one-hot encodes categorical variables, and saves a reference dataset that acts as the baseline for monitoring. The model layer trains a churn classifier, evaluates it with precision, recall, and ROC-AUC, and logs runs to MLflow for reproducibility.

The monitoring layer is the core MLOps piece. Data drift detection compares production features against the reference dataset using PSI and the Kolmogorov-Smirnov (KS) test. Prediction drift monitoring checks whether the distribution of model outputs is shifting over time. Threshold-based alerting converts those statistical signals into JSON reports that can plug into dashboards or incident workflows. Prefect orchestration ties those checks together into a repeatable monitoring pipeline, and FastAPI exposes `/predict` and `/alerts` for programmatic access.

## Key Technical Insights

One important lesson from this project is that data drift and model drift are related, but not identical. A feature distribution can shift while the model remains acceptable for a while, and prediction behavior can shift even when input changes seem modest. That is why both feature-level statistics and output-level monitoring matter.

Another takeaway is that PSI tends to be a practical operational metric for feature drift because it is stable, interpretable, and easy to threshold in monitoring pipelines. KS is still valuable as a complementary statistical test, especially when the goal is to catch distributional change from a different angle. Using both gives a more durable monitoring signal than relying on one test alone.

This project also reinforced that reference dataset design is a serious engineering decision. If the baseline is noisy or outdated, alerts become less trustworthy. Good monitoring starts with choosing the right baseline, not just the right formula.

Finally, alerts are only useful when they are actionable. A drift warning without a runbook is just noise. Structured JSON outputs make follow-up easier for operations teams.

## Architecture Highlights

The architecture is intentionally modular. Feature engineering, training, monitoring, orchestration, and alerting live in separate modules so each part of the lifecycle can evolve independently. MLflow adds experiment traceability, Prefect provides workflow structure, FastAPI exposes system endpoints, Docker makes the environment reproducible, and tests cover API behavior, feature engineering, training, and monitoring logic.

## Real-World Example

Imagine a subscription software business with 100,000 customers. The team trains a churn model in January and deploys it to guide save offers. A few months later, a product change and new competitor shift customer behavior.

Without MLOps, nobody notices until churn has already moved. Leadership sees the KPI drop, the data team scrambles, and retraining becomes a reactive fire drill. With this system, monitoring catches the shift earlier through feature and prediction drift alerts, allowing the team to investigate and retrain before the damage compounds.

## Why I Built This

I built this project to demonstrate the part of machine learning that is often missing from bootcamp-style portfolios: operations. Training a model matters, but keeping that model trustworthy in production is the harder and more valuable skill. This project reflects a practical ML engineering mindset by focusing on monitoring, alerting, orchestration, and reproducibility across the full lifecycle.

## Results & Proof

- [x] Full pipeline from raw data to features, model training, monitoring, and alerting
- [x] MLflow experiment tracking for parameters and evaluation metrics
- [x] Prefect orchestration for repeatable monitoring workflows
- [x] FastAPI endpoints for prediction and alert retrieval
- [x] Drift detection using PSI and KS-based comparisons
- [x] Threshold-based alerting via structured JSON reports
- [x] Dockerized, test-backed project structure for reproducibility

## What's Next

Next steps would include cloud deployment for scheduled retraining, richer dashboards for monitoring visibility, stronger model comparison workflows such as champion-challenger evaluation, and deeper causal analysis to help explain why churn changed, not just that it changed.
