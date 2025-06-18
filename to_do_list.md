# MLOps To-Do List for Model & Pipeline Enhancement

## 1. Add Experiment Tracking & Model Registry with MLflow
- Integrate MLflow tracking into your training pipeline to log parameters, metrics, artifacts, and models.
- Use MLflow Model Registry to version and stage your models (e.g., staging, production).

## 2. Refactor MLflow Code for Production Readiness
- Wrap MLflow tracking code into reusable functions and modules.
- Create clear entry points (e.g., `train.py`, `evaluate.py`) with proper argument parsing.
- Add logging, error handling, and configuration management.

## 3. Incorporate Data Scraping and Dataset Management with PostgreSQL
- Automate scraping, processing, and database loading as modular functions or scripts.
- Maintain versioning or snapshotting of datasets in PostgreSQL for reproducibility.

## 4. Implement Airflow DAG for End-to-End Pipeline Automation
- Build an Airflow DAG to orchestrate:
  - Data scraping and ingestion into PostgreSQL
  - Data preprocessing and feature engineering
  - Model training with MLflow logging
  - Model evaluation and promotion to registry
- Add error alerts and retries.

## 5. Add Experiment Tracking Testing and Validation in the Airflow DAG
- Include tasks that validate MLflow logs and metrics after training.
- Automate performance comparison between new and production models (optional).

## 6. Complete Week 5 Lectures (First Half)
- Watch and summarize key takeaways from the first half of week 5 lectures.

## 7. Complete Week 5 Lectures (Second Half)
- Finish and reflect on the remaining week 5 lectures.

## 8. Deploy Model via MLflow Model Registry
- Deploy production-ready model from MLflow Registry using a serving platform (REST API, Docker, cloud).
- Set up monitoring and logging for the deployed model.
