"""
initial_training_dag - run once, manually.

Validates the raw data, preprocesses all diseases, trains the three candidate
algorithms per disease, registers the champions and creates a first clean batch.
The work runs as a single serialised job in the API service.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from platform_client import call, wait_for_job

try:  # Airflow 3
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2
    from airflow.operators.python import PythonOperator


def run_setup():
    job = call("POST", "/api/setup")
    result = wait_for_job(job, timeout_seconds=3600)
    for disease, info in result["trained"].items():
        print(f"{disease}: {info['champion_algorithm']} v{info['version']} test AUC {info['test_auc']}")


with DAG(
    dag_id="initial_training_dag",
    description="One-time setup: validate, preprocess and train all disease models",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops_platform", "retries": 0},
    tags=["mlops", "setup"],
) as dag:
    PythonOperator(task_id="setup_platform", python_callable=run_setup)
