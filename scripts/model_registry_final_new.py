import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow import register_model

# Add project root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Constants and helpers
from config.constants import MLFLOW_TRACKING_URI, EXPERIMENT_NAME3, DATA_PATH, SEED
from train_optuna_hyperparameter_mlflow_reviews_competitor_pricing_change_criterion_mean_absolute import load_and_preprocess_data

def main():
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME3)

    # Load trained pipeline
    pipeline_path = "models/discount_model_pipeline.pkl"
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Trained pipeline not found at {pipeline_path}")
    
    pipeline = joblib.load(pipeline_path)

    # Load and prepare data
    X, y, *_ = load_and_preprocess_data(DATA_PATH)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Evaluate
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Evaluation metrics on test set: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    regressor = pipeline.named_steps['model']
    regressor_name = regressor.__class__.__name__
    regressor_params = regressor.get_params()

    with mlflow.start_run(run_name="register_trained_model") as run:
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("regressor_type", regressor_name)

        for k, v in regressor_params.items():
            mlflow.log_param(f"param_{k}", str(v))

        # Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Log artifacts (transformers)
        artifact_paths = [
            "models/tfidf_vectorizer.pkl",
            "models/svd_transform.pkl",
            "models/mlb_genres.pkl",
            "models/mlb_tags.pkl",
            "models/competitor_pricing_transformer.pkl"
        ]

        for path in artifact_paths:
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="transformers")
            else:
                print(f"Missing artifact: {path}")

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        print(f"Registering model from: {model_uri}")
        result = register_model(model_uri=model_uri, name="DiscountPredictionModel_MAE")
        print(f"Model registered as version: {result.version}")

    print("Model pipeline evaluated, logged, and registered successfully.")

if __name__ == "__main__":
    main()
