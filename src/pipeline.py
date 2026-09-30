"""
High-level operations shared by the CLI scripts, the API and Airflow.

    setup()                 validate -> preprocess -> train every disease -> first clean batch
    monitor_and_heal(d)     monitor newest batch -> retrain if needed
"""

from __future__ import annotations

from common import config


def setup(create_first_batch: bool = True) -> dict:
    from data_pipeline.create_batches import create_batch
    from data_pipeline.preprocessing import run_all
    from training.train import run_training

    splits = run_all()
    trained = {d: run_training(d) for d in config.diseases()}
    batches = {}
    if create_first_batch:
        for i, d in enumerate(config.diseases()):
            batches[d] = create_batch(d, seed=100 + i)
    return {"splits": splits, "trained": trained, "batches": batches}


def monitor_and_heal(disease: str, force: bool = False) -> dict:
    from monitoring.monitor import run_monitoring
    from retraining.retrain_pipeline import run_retraining

    result = {"monitoring": run_monitoring(disease, force=force), "retraining": None}
    if result["monitoring"]["retraining_needed"]:
        result["retraining"] = run_retraining(disease, result["monitoring"]["reasons"])
    return result
