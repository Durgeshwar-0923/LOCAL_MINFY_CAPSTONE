from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def retrain_task():
    print("Retraining model step executed.")

with DAG(
    dag_id="retrain_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
) as dag:
    retrain = PythonOperator(
        task_id="retrain",
        python_callable=retrain_task,
    )
