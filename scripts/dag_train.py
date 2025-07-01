import pandas as pd
import numpy as np
import joblib
import os
import sys
import mlflow
import mlflow.sklearn
import optuna
import logging

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.constants import MLFLOW_TRACKING_URI, EXPERIMENT_NAME3, DATA_PATH, SEED
from config.preprocessing import get_preprocessor, NUMERIC_FEATURES
from config.hyperparams import get_search_space

logging.basicConfig(level=logging.INFO)


def parse_price(val):
    try:
        return float(str(val).replace('$', '').replace('USD', '').strip())
    except:
        return np.nan


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(
        subset=[
            'total_reviews', 'positive_percent', 'genres', 'tags',
            'current_price', 'discounted_price', 'owners_log_mean', 'days_after_publish'
        ])
    df['current_price'] = df['current_price'].apply(parse_price)
    df['discounted_price'] = df['discounted_price'].apply(parse_price)
    df = df.dropna(subset=['current_price', 'discounted_price'])

    df['discount_pct'] = 1 - (df['discounted_price'] / df['current_price'])
    df['genres'] = df['genres'].fillna('').apply(lambda x: [g.strip() for g in str(x).split(',')])
    df['tags'] = df['tags'].fillna('').apply(lambda x: [t.strip() for t in str(x).split(';')])

    mlb_genres = MultiLabelBinarizer()
    mlb_tags = MultiLabelBinarizer()

    genres_encoded = pd.DataFrame(mlb_genres.fit_transform(df['genres']), columns=mlb_genres.classes_)
    tags_encoded = pd.DataFrame(mlb_tags.fit_transform(df['tags']), columns=mlb_tags.classes_)

    df_enc = pd.concat([df.reset_index(drop=True), genres_encoded, tags_encoded], axis=1)
    features = NUMERIC_FEATURES + list(genres_encoded.columns) + list(tags_encoded.columns)

    X = df_enc[features]
    y = df_enc['discount_pct']

    os.makedirs('models', exist_ok=True)
    joblib.dump((X, y, mlb_genres, mlb_tags), 'models/preprocessed_data.pkl')


def train_and_log_model():
    # Set MLflow tracking URI explicitly here (must match Airflow volume mount)
    mlflow.set_tracking_uri("file:///opt/airflow/mlruns")

    # Create experiment if it does not exist
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME3)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME3)
        logging.info(f"Created MLflow experiment '{EXPERIMENT_NAME3}' with id {experiment_id}")
    else:
        logging.info(f"Using existing MLflow experiment '{EXPERIMENT_NAME3}' with id {experiment.experiment_id}")

    mlflow.set_experiment(EXPERIMENT_NAME3)
    mlflow.sklearn.autolog()

    X, y, mlb_genres, mlb_tags = joblib.load('models/preprocessed_data.pkl')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    preprocessor = get_preprocessor()

    def objective(trial):
        params = get_search_space(trial)
        model_type = params.pop("model_type")

        if model_type == "RandomForest":
            model = RandomForestRegressor(n_estimators=params["rf_n_estimators"], max_depth=params["rf_max_depth"],
                                          min_samples_split=params["rf_min_samples_split"], random_state=SEED, n_jobs=-1)
        elif model_type == "LightGBM":
            model = LGBMRegressor(**params, random_state=SEED, n_jobs=-1)
        elif model_type == "ExtraTrees":
            model = ExtraTreesRegressor(n_estimators=params["et_n_estimators"], max_depth=params["et_max_depth"],
                                       min_samples_split=params["et_min_samples_split"], random_state=SEED, n_jobs=-1)
        elif model_type == "LinearSVR":
            model = LinearSVR(epsilon=params["svr_epsilon"], C=params["svr_C"], max_iter=10000, random_state=SEED)

        pipeline = Pipeline([('preprocess', preprocessor), ('model', model)])

        with mlflow.start_run(nested=True):
            mlflow.set_tag("developer", "F. Kuzey Edes-Huyal")
            mlflow.log_param("model_type", model_type)
            mlflow.log_params(params)
            mlflow.log_param("data_path", DATA_PATH)

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mae = mean_absolute_error(y_val, preds)
            mape = mean_absolute_percentage_error(y_val, preds)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mape", mape)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Optional but clean:
        mlflow.end_run()

        return mae

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    for model_name in ["RandomForest", "LightGBM", "ExtraTrees", "LinearSVR"]:
        study.enqueue_trial({"model_type": model_name})
    study.optimize(objective, n_trials=10)

    best_trial = study.best_trial
    joblib.dump((best_trial.params, X_train, y_train, X_val, y_val, preprocessor, mlb_genres, mlb_tags), 'models/best_model_info.pkl')


def finalize_and_log():
    mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME3)

    best_params, X_train, y_train, X_val, y_val, preprocessor, mlb_genres, mlb_tags = joblib.load('models/best_model_info.pkl')
    best_model_type = best_params.pop("model_type")

    if best_model_type == "RandomForest":
        final_model = RandomForestRegressor(n_estimators=best_params["rf_n_estimators"], max_depth=best_params["rf_max_depth"],
                                            min_samples_split=best_params["rf_min_samples_split"], random_state=SEED, n_jobs=-1)
    elif best_model_type == "LightGBM":
        final_model = LGBMRegressor(**best_params, random_state=SEED, n_jobs=-1)
    elif best_model_type == "ExtraTrees":
        final_model = ExtraTreesRegressor(n_estimators=best_params["et_n_estimators"], max_depth=best_params["et_max_depth"],
                                          min_samples_split=best_params["et_min_samples_split"], random_state=SEED, n_jobs=-1)
    elif best_model_type == "LinearSVR":
        final_model = LinearSVR(epsilon=best_params["svr_epsilon"], C=best_params["svr_C"], max_iter=10000, random_state=SEED)

    pipeline_final = Pipeline([('preprocess', preprocessor), ('model', final_model)])
    pipeline_final.fit(X_train, y_train)
    final_preds = pipeline_final.predict(X_val)

    final_rmse = np.sqrt(mean_squared_error(y_val, final_preds))
    final_mae = mean_absolute_error(y_val, final_preds)
    final_mape = mean_absolute_percentage_error(y_val, final_preds)

    model_path = 'models/discount_model_pipeline_no_review_score_no_competitor.pkl'
    genres_path = 'models/mlb_genres_no_review_score_no_competitor.pkl'
    tags_path = 'models/mlb_tags_no_review_score_no_competitor.pkl'

    joblib.dump(pipeline_final, model_path)
    joblib.dump(mlb_genres, genres_path)
    joblib.dump(mlb_tags, tags_path)

    with mlflow.start_run(run_name="final_model_run"):
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", best_model_type)
        mlflow.log_metric("rmse", final_rmse)
        mlflow.log_metric("mae", final_mae)
        mlflow.log_metric("mape", final_mape)
        mlflow.sklearn.log_model(pipeline_final, "model", input_example=X_val.iloc[:5])

        mlflow.log_artifact(genres_path, artifact_path="transformers")
        mlflow.log_artifact(tags_path, artifact_path="transformers")

    mlflow.end_run()