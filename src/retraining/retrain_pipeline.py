"""
Self-healing retraining for ONE disease.

    1. Historical train + training part of every batch
    2. Train every algorithm, choose the challenger on VALIDATION AUC
    3. Register the challenger (always - rejected ones stay auditable)
    4. Compare against the champion on recent data + the original test set
    5. Promote if better; the drift baseline moves with the new champion

    python src/retraining/retrain_pipeline.py --disease diabetes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import pandas as pd

from common import config
from common.storage import append_jsonl, read_jsonl, utc_now
from data_pipeline.batch_loader import load_for_retraining
from monitoring.alerting import send_alert
from retraining.model_comparator import compare_models
from retraining.promotion import promote_challenger
from training import model_registry
from training.champion_selector import select_champion
from training.train import load_split, train_candidates


def history_path():
    return config.monitoring_dir() / "retraining_history.jsonl"


def run_retraining(disease: str, trigger_reasons: list[str] | None = None) -> dict:
    trigger_reasons = trigger_reasons or ["MANUAL"]
    print(f"\n===== Retraining {disease} (reasons: {trigger_reasons}) =====\n")
    config.set_experiment(f"{disease}_retraining")
    feats = config.features(disease)

    data = load_for_retraining(disease)
    train_df = data["train"]
    X_train, y_train = train_df[feats], train_df["Outcome"]
    X_val, y_val = load_split(disease, "val")
    test_df = pd.read_csv(config.processed_dir(disease) / "test.csv")

    with mlflow.start_run(run_name=f"{disease}_retrain"):
        mlflow.set_tags({"training_mode": "retraining", "trigger": ",".join(trigger_reasons)})
        mlflow.log_param("batches_used", ",".join(data["batches_used"]) or "none")
        mlflow.log_param("train_rows", len(train_df))

        models, val_metrics = train_candidates(X_train, y_train, X_val, y_val)
        challenger_name = select_champion({n: m["auc"] for n, m in val_metrics.items()})
        challenger = models[challenger_name]

        comparison = compare_models(disease, challenger, test_df, data["recent_eval"], trigger_reasons)
        for key in ("champion_test_auc", "challenger_test_auc", "champion_recent_auc", "challenger_recent_auc"):
            if comparison.get(key) is not None:
                mlflow.log_metric(key, comparison[key])

        version = model_registry.register_model(
            disease, challenger, challenger_name, X_train.head(5),
            metrics={"val_auc": val_metrics[challenger_name]["auc"],
                     "test_auc": comparison.get("challenger_test_auc"),
                     "recent_auc": comparison.get("challenger_recent_auc")},
            tags={"training_mode": "retraining", "train_rows": len(train_df),
                  "status": "promoted" if comparison["should_promote"] else "rejected"},
        )

        if comparison["should_promote"]:
            promote_challenger(disease, version, challenger_name, train_df, comparison)
        else:
            send_alert(disease, "CHALLENGER_REJECTED", {
                "challenger_version": version, "algorithm": challenger_name,
                "decision": comparison["decision"],
            })

    record = {
        "timestamp": utc_now(), "disease": disease, "trigger_reasons": trigger_reasons,
        "challenger_version": version, "challenger_algorithm": challenger_name,
        "batches_used": data["batches_used"], "train_rows": len(train_df), **comparison,
    }
    append_jsonl(history_path(), record)
    print(f"\n===== Finished retraining {disease}: promoted={comparison['should_promote']} =====\n")
    return record


def retraining_history(disease: str | None = None, limit: int = 100) -> list[dict]:
    rows = read_jsonl(history_path())
    if disease:
        rows = [r for r in rows if r["disease"] == disease]
    return list(reversed(rows[-limit:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", choices=config.diseases(), required=True)
    run_retraining(parser.parse_args().disease)
