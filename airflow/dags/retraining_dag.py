"""
retraining_dag - triggered by monitoring_dag (or manually with {"disease": "..."}).

max_active_runs=1 queues concurrent triggers instead of running three retrains
against the registry at once. A missing/unknown disease fails loudly instead of
silently retraining diabetes.
"""

from __future__ import annotations

import ast
from datetime import datetime

from airflow import DAG
from platform_client import DISEASES, call, wait_for_job

try:  # Airflow 3
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2
    from airflow.operators.python import PythonOperator


def _reasons(conf: dict) -> list[str]:
    raw = conf.get("reasons") or ["MANUAL"]
    if isinstance(raw, str):  # templated XCom arrives as the string "['DATA_DRIFT']"
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = [raw]
    return [str(r) for r in raw] or ["MANUAL"]


def run_retraining(**context):
    conf = context["dag_run"].conf or {}
    disease = conf.get("disease")
    if disease not in DISEASES:
        raise ValueError(f"Trigger this DAG with conf {{'disease': one of {DISEASES}}}; got {disease!r}")

    reasons = _reasons(conf)
    job = call("POST", f"/api/retraining/{disease}/run?reason={','.join(reasons)}")
    result = wait_for_job(job, timeout_seconds=3600)
    print(f"{disease}: challenger v{result['challenger_version']} promoted={result['should_promote']} "
          f"({result['decision']})")
    return {"disease": disease, "promoted": result["should_promote"]}


with DAG(
    dag_id="retraining_dag",
    description="Retrains one disease model and promotes it only if it beats the champion",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops_platform", "retries": 0},
    tags=["mlops", "retraining"],
) as dag:
    PythonOperator(task_id="retrain_model", python_callable=run_retraining)
