"""
monitoring_dag - every day, for every disease:

    check_<disease>  checks the newest UNCHECKED batch for drift and performance.
                     It is a short-circuit task: it returns True only when
                     retraining is needed, otherwise the trigger below is skipped.
    trigger_retrain_<disease>  starts retraining_dag with {"disease": ..., "reasons": [...]}

(The old DAG triggered retraining unconditionally, always re-checked batch_1 and
imported a monitoring.monitor module that did not exist.)
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from platform_client import DISEASES, call

try:  # Airflow 3
    from airflow.providers.standard.operators.python import ShortCircuitOperator
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
except ImportError:  # Airflow 2
    from airflow.operators.python import ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator


def check_disease(disease: str, **context) -> bool:
    # heal=false: Airflow, not the API, decides to start the retraining DAG
    result = call("POST", f"/api/monitoring/{disease}/run?heal=false")["monitoring"]
    print(result)
    context["ti"].xcom_push(key="reasons", value=result.get("reasons", []))
    return bool(result.get("retraining_needed"))


with DAG(
    dag_id="monitoring_dag",
    description="Checks new batches for drift / performance drops and triggers retraining",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops_platform", "retries": 1},
    tags=["mlops", "monitoring"],
) as dag:
    for disease in DISEASES:
        check = ShortCircuitOperator(
            task_id=f"check_{disease}",
            python_callable=check_disease,
            op_kwargs={"disease": disease},
        )
        trigger = TriggerDagRunOperator(
            task_id=f"trigger_retrain_{disease}",
            trigger_dag_id="retraining_dag",
            conf={
                "disease": disease,
                "reasons": f"{{{{ ti.xcom_pull(task_ids='check_{disease}', key='reasons') }}}}",
            },
        )
        check >> trigger
