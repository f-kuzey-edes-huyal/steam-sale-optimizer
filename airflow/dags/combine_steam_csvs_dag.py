from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
sys.path.append('/opt/airflow/scripts')

from load_and_combine import load_csv_to_postgres_and_export

with DAG(
    dag_id='secure_combine_steam_csvs',
    schedule_interval='@once',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['secure', 'steam'],
) as dag:

    combine_task = PythonOperator(
        task_id='load_combine_export',
        python_callable=load_csv_to_postgres_and_export,
    )

    combine_task
