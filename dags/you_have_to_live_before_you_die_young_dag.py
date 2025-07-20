from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests, os, joblib, time
import sys

# Add script path for your existing imports
sys.path.append('/opt/airflow/scripts')

# === Step 1: Scraping & Combining ===
from airflow_main_scraper1_new import scrape_steam_data
from load_and_combine_new import load_csv_to_postgres_and_export

# === Step 2: Training & Registering Model ===
from train_model import train_model
from register_model import register_model


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 7, 17),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# === Integration / Deployment steps ===

def check_model_file():
    model_path = "/opt/airflow/models/discount_model_pipeline.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully:", model)
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")

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


with DAG(
    dag_id='you_have_to_live_before_you_die_young_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='Full pipeline: Scrape, Train, Register, Deploy and Integration Test FastAPI',
    tags=['orchestration', 'mlflow', 'steam', 'fastapi'],
) as dag:

    # Existing pipeline tasks
    scrape_task = PythonOperator(
        task_id='scrape_steam_data',
        python_callable=scrape_steam_data,
    )

    combine_task = PythonOperator(
        task_id='combine_and_export_csv',
        python_callable=load_csv_to_postgres_and_export,
    )

    train_task = PythonOperator(
        task_id='train_discount_model',
        python_callable=train_model,
    )

    register_task = PythonOperator(
        task_id='register_discount_model',
        python_callable=register_model,
    )

    # Integration & Deployment tasks
    test_model_artifact = PythonOperator(
        task_id="integration_test_model_file",
        python_callable=check_model_file,
    )

    deploy_model_task = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
    )

    test_fastapi_task = PythonOperator(
        task_id='test_fastapi_prediction',
        python_callable=test_fastapi_prediction,
    )


    # Task dependencies: Full pipeline + integration steps in order
    scrape_task >> combine_task >> train_task >> register_task >> test_model_artifact >> deploy_model_task >> test_fastapi_task
