from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def print_welcome():
    print('Welcome to Airflow!')

def print_date():
    print('Today is {}'.format(datetime.today().date()))

def print_motivation():
    print("Keep pushing forward! Every day is a new opportunity to learn and grow.")

default_args = {
    'start_date': datetime.now() - timedelta(days=1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='welcome_dag',
    default_args=default_args,
    schedule='0 23 * * *',
    catchup=False,
    tags=['example']
) as dag:

    print_welcome_task = PythonOperator(
        task_id='print_welcome',
        python_callable=print_welcome,
    )

    print_date_task = PythonOperator(
        task_id='print_date',
        python_callable=print_date,
    )

    print_motivation_task = PythonOperator(
        task_id='print_motivation',
        python_callable=print_motivation,
    )

    print_welcome_task >> print_date_task >> print_motivation_task
