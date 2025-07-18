from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import Python callables from outside the DAG folder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from train_model import train_model
from register_model import register_model

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='discount_model_training_pipeline_new',
    default_args=default_args,
    description='Train discount prediction model and register it to MLflow',
    schedule_interval=None,  # You can change to '@daily', '@weekly', etc.
    start_date=datetime(2025, 7, 17),
    catchup=False,
    tags=['mlflow', 'discount', 'optuna', 'registry']
) as dag:

    train = PythonOperator(
        task_id='train_discount_model',
        python_callable=train_model
    )

    register = PythonOperator(
        task_id='register_trained_model',
        python_callable=register_model
    )

    train >> register  # Train first, then register