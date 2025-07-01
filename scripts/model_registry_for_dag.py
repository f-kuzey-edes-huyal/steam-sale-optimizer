import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Add root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import constants and data loading function
from config.constants import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, DATA_PATH, SEED
from train_optuna_hyperparameter_mlflow_reviews_competitor_pricing_change_criterion_mean_absolute import load_and_preprocess_data


def evaluate_and_register_model():
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load trained pipeline
    pipeline_path = "models/discount_model_pipeline.pkl"
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Trained model pipeline not found at {pipeline_path}")
    pipeline_final = joblib.load(pipeline_path)

    # Load and split data
    X, y, *_ = load_and_preprocess_data(DATA_PATH)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Evaluate
    preds = pipeline_final.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Extract regressor and its params
    regressor = pipeline_final.named_steps['model']
    regressor_name = regressor.__class__.__name__
    regressor_params = regressor.get_params()

    with mlflow.start_run(run_name="register_already_trained_model") as run:
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("regressor_type", regressor_name)

        for param_name, param_value in regressor_params.items():
            mlflow.log_param(f"param_{param_name}", str(param_value))

        mlflow.sklearn.log_model(pipeline_final, artifact_path="model")

        # Log additional preprocessing transformers
        transformer_files = [
            "tfidf_vectorizer.pkl",
            "svd_transform.pkl",
            "mlb_genres.pkl",
            "mlb_tags.pkl",
            "competitor_pricing_transformer.pkl"
        ]

        for file_name in transformer_files:
            full_path = os.path.join("models", file_name)
            if os.path.exists(full_path):
                mlflow.log_artifact(full_path, artifact_path="transformers")

        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name="DiscountPredictionModel_MAE")

        print(f"Registered model version: {result.version}")

    print("Model registered successfully.")
