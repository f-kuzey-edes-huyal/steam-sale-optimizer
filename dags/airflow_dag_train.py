from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Add scripts folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from dag_train import load_data, train_and_log_model, finalize_and_log
from task_registry_stub import evaluate_and_register_model

default_args = {
    'owner': 'kuzey',
    'start_date': datetime(2024, 1, 1),
    'retries': 1
}

with DAG(
    dag_id='discount_model_training_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["mlflow", "training", "optuna"]
) as dag:

    load_data_task = PythonOperator(
        task_id='load_and_preprocess_data',
        python_callable=load_data
    )

    train_model_task = PythonOperator(
        task_id='train_and_log_models',
        python_callable=train_and_log_model
    )

    finalize_model_task = PythonOperator(
        task_id='finalize_and_log_model',
        python_callable=finalize_and_log
    )

    #register_model_task = PythonOperator(
       # task_id='evaluate_and_register_model',
       # python_callable=evaluate_and_register_model
    #)

    #load_data_task >> train_model_task >> finalize_model_task >> register_model_task

    load_data_task >> train_model_task >> finalize_model_task 
