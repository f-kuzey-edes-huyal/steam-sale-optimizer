from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
sys.path.append('/opt/airflow/scripts')

from airflow_main_scraper1 import scrape_steam_data
from load_and_combine import load_csv_to_postgres_and_export

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
    dag_id='scrape_and_combine_steam_csvs_dag',
    default_args=default_args,
    schedule_interval='@once',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['secure', 'steam'],
) as dag:

    scrape_task = PythonOperator(
        task_id='scrape_steam_data',
        python_callable=scrape_steam_data,
    )

    combine_task = PythonOperator(
        task_id='load_combine_export',
        python_callable=load_csv_to_postgres_and_export,
    )

    scrape_task >> combine_task
