from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
import time

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
}

def deploy_model():
    url = "http://fastapi-app:80/reload_model"
    for attempt in range(5):
        try:
            response = requests.post(url, timeout=5)
            print(f"Attempt {attempt + 1}: Status {response.status_code}, Response: {response.json()}")
            if response.status_code == 200:
                return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise Exception("FastAPI reload_model endpoint failed after retries")


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
            print(f"Attempt {attempt + 1}: Status {response.status_code}, Response: {response.json()}")
            if response.status_code == 200:
                return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise Exception("FastAPI predict endpoint failed after retries")


with DAG(
    "deploy_and_test_fastapi_model",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description="Deploys model and tests FastAPI prediction endpoint",
) as dag:

    deploy_model_task = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model
    )

    test_fastapi = PythonOperator(
        task_id="test_fastapi_prediction",
        python_callable=test_fastapi_prediction
    )

    deploy_model_task >> test_fastapi
