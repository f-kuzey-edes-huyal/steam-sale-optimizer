from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import time
import sys
import os

# Add script path
sys.path.append('/opt/airflow/scripts')

# === Step 1: Scraping & Combining ===
from airflow_main_scraper1_new import scrape_steam_data
from load_and_combine_new import load_csv_to_postgres_and_export

# === Step 2: Training & Registering Model ===
from train_model import train_model
from register_model import register_model

# === Step 3: FastAPI Test ===
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
                return
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    raise Exception("FastAPI endpoint did not respond after retries")


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 7, 17),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='you_have_to_live_before_you_die_young_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='End-to-end pipeline: Scrape, Train, Register, Test Prediction',
    tags=['orchestration', 'mlflow', 'steam', 'fastapi'],
) as dag:

    # Step 1
    scrape_task = PythonOperator(
        task_id='scrape_steam_data',
        python_callable=scrape_steam_data,
    )

    combine_task = PythonOperator(
        task_id='combine_and_export_csv',
        python_callable=load_csv_to_postgres_and_export,
    )

    # Step 2
    train_task = PythonOperator(
        task_id='train_discount_model',
        python_callable=train_model,
    )

    register_task = PythonOperator(
        task_id='register_discount_model',
        python_callable=register_model,
    )

    # Step 3
    test_fastapi_task = PythonOperator(
        task_id='test_fastapi_prediction',
        python_callable=test_fastapi_prediction,
    )

    # Task Dependencies
    scrape_task >> combine_task >> train_task >> register_task >> test_fastapi_task
