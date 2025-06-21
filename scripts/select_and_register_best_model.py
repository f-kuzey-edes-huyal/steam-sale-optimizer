import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.constants import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, DATA_PATH, SEED
from config.preprocessing import get_preprocessor, NUMERIC_FEATURES

# Set MLflow tracking and experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Helper function to parse price values like '$19.99' or 'USD 9.99'
def parse_price(val):
    try:
        return float(str(val).replace('$', '').replace('USD', '').strip())
    except:
        return np.nan

# Preprocessing function to clean and prepare the dataset
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Drop rows with missing important columns
    df = df.dropna(subset=[
        'total_reviews', 'positive_percent', 'genres', 'tags',
        'current_price', 'discounted_price', 'owners_log_mean', 'days_after_publish'
    ])

    # Convert string price values to float
    df['current_price'] = df['current_price'].apply(parse_price)
    df['discounted_price'] = df['discounted_price'].apply(parse_price)

    # Remove rows with missing or invalid price data
    df = df.dropna(subset=['current_price', 'discounted_price'])

    # Compute target: discount percentage
    df['discount_pct'] = 1 - (df['discounted_price'] / df['current_price'])

    # Convert genre and tag strings to lists
    df['genres'] = df['genres'].fillna('').apply(lambda x: [g.strip() for g in str(x).split(',')])
    df['tags'] = df['tags'].fillna('').apply(lambda x: [t.strip() for t in str(x).split(';')])

    # Load MultiLabelBinarizers to encode genres and tags (previously fitted)
    mlb_genres = joblib.load("models/mlb_genres.pkl")
    mlb_tags = joblib.load("models/mlb_tags.pkl")

    # Encode genre and tag lists into one-hot vectors
    genres_encoded = pd.DataFrame(mlb_genres.transform(df['genres']), columns=mlb_genres.classes_)
    tags_encoded = pd.DataFrame(mlb_tags.transform(df['tags']), columns=mlb_tags.classes_)

    # Combine all features into a single DataFrame
    df_enc = pd.concat([df.reset_index(drop=True), genres_encoded, tags_encoded], axis=1)

    # Final feature matrix and target variable
    features = NUMERIC_FEATURES + list(genres_encoded.columns) + list(tags_encoded.columns)
    X = df_enc[features]
    y = df_enc['discount_pct']
    return X, y

# Load dataset and split into training and test sets
X, y = load_and_preprocess_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
preprocessor = get_preprocessor()

# Setup MLflow client to retrieve previous runs
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

# Retrieve all past runs sorted by RMSE (ascending)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    run_view_type=1,  # only active runs
    max_results=100,
    order_by=["metrics.rmse ASC"]
)

# Filter out runs that don't have an RMSE logged
runs = [r for r in runs if "rmse" in r.data.metrics]
if not runs:
    raise ValueError("No past runs with 'rmse' metric found in the experiment.")

# Select the run with the best (lowest) RMSE
best_run = runs[0]
best_run_id = best_run.info.run_id
best_model_uri = f"runs:/{best_run_id}/model"

print(f"Loading best model from run: {best_run_id}")

# Load the best model pipeline (includes preprocessing and model)
best_model = mlflow.sklearn.load_model(best_model_uri)

# Evaluate the model on the hold-out test set
preds = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Final evaluation on test set: RMSE = {rmse:.4f}")

# Register the model in MLflow Model Registry
model_name = "DiscountPredictionModel"
result = mlflow.register_model(model_uri=best_model_uri, name=model_name)
print(f"Model registered as '{model_name}' with version: {result.version}")

# Save the model locally for serving or deployment
joblib.dump(best_model, "models/best_model_pipeline.pkl")
print("Best model pipeline saved to 'models/best_model_pipeline.pkl'")
