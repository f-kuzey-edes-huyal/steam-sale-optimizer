from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
import time

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
}

def test_fastapi_prediction():
    # Use host and port from your compose file (localhost:8082 if from host machine)
    # If Airflow runs in Docker and shares network with fastapi-app, use service name and internal port (fastapi-app:80)
    
    url = "http://fastapi-app:80/predict"  # inside Docker network
    # If testing from Airflow host or outside Docker network, use "http://localhost:8082/predict"
    
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

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=test_data, timeout=5)
            print(f"Attempt {attempt+1}: Status {response.status_code}, Response: {response.json()}")
            if response.status_code == 200:
                return
            else:
                raise Exception(f"Unexpected status code: {response.status_code}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    raise Exception("FastAPI endpoint is not responding after retries")

with DAG(
    "test_fastapi_service",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description="Test FastAPI prediction endpoint",
) as dag:

    test_fastapi = PythonOperator(
        task_id="test_fastapi_prediction",
        python_callable=test_fastapi_prediction
    )


    test_fastapi