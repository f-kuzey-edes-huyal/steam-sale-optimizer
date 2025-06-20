import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
from lightgbm import LGBMRegressor
import joblib
import mlflow
import mlflow.sklearn
import optuna

def parse_price(val):
    try:
        return float(str(val).replace('$', '').replace('USD', '').strip())
    except:
        return np.nan

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=[
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

    features = [
        'total_reviews', 'positive_percent', 'current_price',
        'owners_log_mean', 'days_after_publish'
    ] + list(genres_encoded.columns) + list(tags_encoded.columns)

    X = df_enc[features]
    y = df_enc['discount_pct']
    return X, y, mlb_genres, mlb_tags

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Grizz does experiment tracking")
mlflow.sklearn.autolog()

X, y, mlb_genres, mlb_tags = load_and_preprocess_data("data/combined4.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = ['total_reviews', 'positive_percent', 'current_price', 'owners_log_mean', 'days_after_publish']
num_transformer = Pipeline([('scaler', StandardScaler())])
preprocessor = ColumnTransformer([('num', num_transformer, num_features)], remainder='passthrough')

used_models = set()

def objective(trial):
    model_name = trial.suggest_categorical("model_type", ["RandomForest", "LightGBM", "ExtraTrees", "LinearSVR"])
    used_models.add(model_name)
    params = {}

    if model_name == "RandomForest":
        params['rf_n_estimators'] = trial.suggest_int("rf_n_estimators", 50, 500, step=25)
        params['rf_max_depth'] = trial.suggest_int("rf_max_depth", 5, 50)
        params['rf_min_samples_split'] = trial.suggest_int("rf_min_samples_split", 2, 20)
        model = RandomForestRegressor(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_split=params["rf_min_samples_split"],
            random_state=42, n_jobs=-1
        )

    elif model_name == "LightGBM":
        params['lgbm_n_estimators'] = trial.suggest_int("lgbm_n_estimators", 50, 500, step=25)
        params['lgbm_max_depth'] = trial.suggest_int("lgbm_max_depth", 3, 30)
        params['lgbm_learning_rate'] = trial.suggest_float("lgbm_learning_rate", 0.001, 0.3, log=True)
        params['lgbm_num_leaves'] = trial.suggest_int("lgbm_num_leaves", 20, 100)
        model = LGBMRegressor(
            n_estimators=params["lgbm_n_estimators"],
            max_depth=params["lgbm_max_depth"],
            learning_rate=params["lgbm_learning_rate"],
            num_leaves=params["lgbm_num_leaves"],
            random_state=42, n_jobs=-1
        )

    elif model_name == "ExtraTrees":
        params['et_n_estimators'] = trial.suggest_int("et_n_estimators", 50, 500, step=25)
        params['et_max_depth'] = trial.suggest_int("et_max_depth", 5, 50)
        params['et_min_samples_split'] = trial.suggest_int("et_min_samples_split", 2, 20)
        model = ExtraTreesRegressor(
            n_estimators=params["et_n_estimators"],
            max_depth=params["et_max_depth"],
            min_samples_split=params["et_min_samples_split"],
            random_state=42, n_jobs=-1
        )

    elif model_name == "LinearSVR":
        params['svr_epsilon'] = trial.suggest_float("svr_epsilon", 1e-4, 0.1, log=True)
        params['svr_C'] = trial.suggest_float("svr_C", 0.01, 100, log=True)
        model = LinearSVR(
            epsilon=params["svr_epsilon"],
            C=params["svr_C"],
            max_iter=10000, random_state=42
        )

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])

    with mlflow.start_run(nested=True):
        mlflow.set_tag("developer", "F. Kuzey Edes-Huyal")
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(params)
        mlflow.log_param("data_path", "data/combined4.csv")
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mlflow.log_metric("rmse", rmse)

    return rmse

# Study setup
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
for model_name in ["RandomForest", "LightGBM", "ExtraTrees", "LinearSVR"]:
    study.enqueue_trial({"model_type": model_name})
study.optimize(objective, n_trials=40)

# Results
print("Best trial:")
trial = study.best_trial
print(f"  RMSE: {trial.value}")
print("  Params:")
for k, v in trial.params.items():
    print(f"    {k}: {v}")

print("\nModels tried during optimization:")
for m in sorted(used_models):
    print(f" - {m}")

with mlflow.start_run(run_name="summary_of_models_used", nested=True):
    mlflow.log_param("models_tried", ", ".join(sorted(used_models)))
    mlflow.set_tag("models_tried", ", ".join(sorted(used_models)))

# Final model reconstruction
best_params = trial.params.copy()
best_model_type = best_params.pop("model_type")

if best_model_type == "RandomForest":
    model_params = {
        "n_estimators": best_params["rf_n_estimators"],
        "max_depth": best_params["rf_max_depth"],
        "min_samples_split": best_params["rf_min_samples_split"]
    }
    final_model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1)

elif best_model_type == "LightGBM":
    final_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1)

elif best_model_type == "ExtraTrees":
    model_params = {
        "n_estimators": best_params["et_n_estimators"],
        "max_depth": best_params["et_max_depth"],
        "min_samples_split": best_params["et_min_samples_split"]
    }
    final_model = ExtraTreesRegressor(**model_params, random_state=42, n_jobs=-1)

elif best_model_type == "LinearSVR":
    final_model = LinearSVR(**best_params, max_iter=10000, random_state=42)

# Final pipeline + eval
pipeline_final = Pipeline([
    ('preprocess', preprocessor),
    ('model', final_model)
])
pipeline_final.fit(X_train, y_train)
final_preds = pipeline_final.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))

print(f"Final model ({best_model_type}) RMSE: {final_rmse:.4f}")

with mlflow.start_run(run_name="final_model_run"):
    mlflow.log_params(best_params)
    mlflow.log_param("model_type", best_model_type)
    mlflow.log_metric("rmse", final_rmse)
    mlflow.sklearn.log_model(pipeline_final, "model", input_example=X_test.iloc[:5])

# Save final pipeline + encoders
joblib.dump(pipeline_final, 'models/discount_model_pipeline.pkl')
joblib.dump(mlb_genres, 'models/mlb_genres.pkl')
joblib.dump(mlb_tags, 'models/mlb_tags.pkl')
