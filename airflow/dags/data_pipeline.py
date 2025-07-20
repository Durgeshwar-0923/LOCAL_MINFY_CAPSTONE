from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def dummy_task():
    print("Data pipeline step executed.")

with DAG(
    dag_id="data_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    run_step = PythonOperator(
        task_id="run_step",
        python_callable=dummy_task,
    )
