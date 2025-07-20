from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import requests, os, joblib, time

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

# Step: Integration test after model registration
def check_model_file():
    model_path = "/opt/airflow/models/discount_model_pipeline.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully:", model)
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")

# Step: Deploy FastAPI model (trigger reload)
def deploy_model():
    url = "http://fastapi-app:80/reload_model"
    for attempt in range(5):
        try:
            response = requests.post(url, timeout=5)
            print(f"Attempt {attempt+1}: Status {response.status_code}, Response: {response.json()}")
            if response.status_code == 200:
                return
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    raise Exception("FastAPI reload_model endpoint failed after retries")

# Step: Integration test on prediction
def test_fastapi_prediction():
    url = "http://fastapi-app:80/predict"
    test_data = {
        "game_id": 123,
        "name": "Test Game",
        "release_date": "2023-01-01",
        "total_reviews": 1000,
        "positive_percent": 75,
        "genres": "Action, Adventure",
        "tags": "Multiplayer;Co-op",
        "current_price": "$20.00",
        "discounted_price": "$10.00",
        "owners": "50000..100000",
        "days_after_publish": 365,
        "review": "Great game, loved it!",
        "owner_min": 50000,
        "owner_max": 100000,
        "owners_log_mean": 11.5
    }

    for attempt in range(5):
        try:
            response = requests.post(url, json=test_data, timeout=5)
            print(f"Attempt {attempt+1}: Status {response.status_code}, Response: {response.json()}")
            if response.status_code == 200:
                prediction = response.json().get("predicted_discount_pct")
                if prediction is None:
                    raise Exception("Missing prediction in response")
                return
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    raise Exception("FastAPI predict endpoint failed after retries")

# --- DAG Definition ---
with DAG(
    "deploy_and_test_fastapi_model_integration",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description="Deploy model and perform integration tests",
) as dag:

    test_model_artifact = PythonOperator(
        task_id="integration_test_model_file",
        python_callable=check_model_file,
    )

    deploy_model_task = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
    )

    test_fastapi = PythonOperator(
        task_id="test_fastapi_prediction",
        python_callable=test_fastapi_prediction
    )

    # Task dependencies
    test_model_artifact >> deploy_model_task >> test_fastapi
