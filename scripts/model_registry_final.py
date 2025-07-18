import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Add project root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your data loading and preprocessing functions/constants
from config.constants import MLFLOW_TRACKING_URI_local, EXPERIMENT_NAME, DATA_PATH, SEED
from train_optuna_hyperparameter_mlflow_reviews_competitor_pricing_change_criterion_mean_absolute import load_and_preprocess_data

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_local)
mlflow.set_experiment(EXPERIMENT_NAME)

# Load saved pipeline (already trained)
pipeline_final = joblib.load("models/discount_model_pipeline.pkl")

# Load and preprocess data (to get test data for evaluation)
X, y, *_ = load_and_preprocess_data(DATA_PATH)

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Predict on test set
preds = pipeline_final.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"Evaluation metrics on test set: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

# Extract regressor from pipeline
regressor = pipeline_final.named_steps['model']
regressor_name = regressor.__class__.__name__
regressor_params = regressor.get_params()

with mlflow.start_run(run_name="register_already_trained_model") as run:
    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    # Log regressor type
    mlflow.log_param("regressor_type", regressor_name)

    # Log all regressor parameters
    for param_name, param_value in regressor_params.items():
        # Convert non-string params to string for logging
        mlflow.log_param(f"param_{param_name}", str(param_value))

    # Log the pipeline model
    mlflow.sklearn.log_model(pipeline_final, artifact_path="model")

    # Log transformers (must exist on disk)
    mlflow.log_artifact("models/tfidf_vectorizer.pkl", artifact_path="transformers")
    mlflow.log_artifact("models/svd_transform.pkl", artifact_path="transformers")
    mlflow.log_artifact("models/mlb_genres.pkl", artifact_path="transformers")
    mlflow.log_artifact("models/mlb_tags.pkl", artifact_path="transformers")
    mlflow.log_artifact("models/competitor_pricing_transformer.pkl", artifact_path="transformers")

    # Register the model
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name="DiscountPredictionModel_MAE")

    print(f"Registered model version: {result.version}")

print("Model pipeline evaluated, logged, and registered successfully.")
