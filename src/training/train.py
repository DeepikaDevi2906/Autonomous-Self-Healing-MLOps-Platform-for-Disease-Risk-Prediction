"""
Initial training for ONE disease.

    1. Load processed train / val / test
    2. Train every algorithm in training.models.TRAINERS
    3. Choose the champion on VALIDATION AUC
    4. Report its TEST metrics (used later as the performance baseline)
    5. Register it (with its preprocessor) and make it the champion

    python src/training/train.py --disease diabetes
    python src/training/train.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import pandas as pd

from common import config
from common.storage import append_jsonl, utc_now
from training import model_registry
from training.champion_selector import select_champion
from training.evaluation import evaluate
from training.models import TRAINERS


def load_split(disease: str, split: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(config.processed_dir(disease) / f"{split}.csv")
    return df.drop(columns=["Outcome"]), df["Outcome"]


def train_candidates(X_train, y_train, X_val, y_val) -> tuple[dict, dict]:
    models, val_metrics = {}, {}
    for name, trainer in TRAINERS.items():
        print(f"Training {name}...")
        models[name] = trainer(X_train, y_train)
        val_metrics[name] = evaluate(models[name], X_val, y_val)
    return models, val_metrics


def run_training(disease: str) -> dict:
    print(f"\n===== Training models for: {disease} =====\n")
    config.set_experiment(disease)

    X_train, y_train = load_split(disease, "train")
    X_val, y_val = load_split(disease, "val")
    X_test, y_test = load_split(disease, "test")
    print(f"Loaded {disease}: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test rows")

    with mlflow.start_run(run_name=f"{disease}_initial"):
        mlflow.set_tag("training_mode", "initial")
        models, val_metrics = train_candidates(X_train, y_train, X_val, y_val)
        for name, m in val_metrics.items():
            mlflow.log_metric(f"{name}_val_auc", m["auc"])

        champion_name = select_champion({n: m["auc"] for n, m in val_metrics.items()})
        champion = models[champion_name]
        test_metrics = evaluate(champion, X_test, y_test)
        mlflow.log_param("champion", champion_name)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if v is not None})

        version = model_registry.register_model(
            disease, champion, champion_name, X_train.head(5),
            metrics={"val_auc": val_metrics[champion_name]["auc"],
                     **{f"test_{k}": v for k, v in test_metrics.items()}},
            tags={"training_mode": "initial", "train_rows": len(X_train)},
        )
        model_registry.promote(disease, version, reason="initial training")

    summary = {
        "timestamp": utc_now(), "disease": disease, "event": "initial_training",
        "champion_algorithm": champion_name, "version": version,
        "val_auc": val_metrics[champion_name]["auc"], "test_auc": test_metrics["auc"],
        "candidates": {n: m["auc"] for n, m in val_metrics.items()},
    }
    append_jsonl(config.monitoring_dir() / "training_history.jsonl", summary)
    print(f"\n===== Finished training {disease}: {champion_name} v{version}, test AUC {test_metrics['auc']} =====\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", choices=config.diseases())
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.disease:
        parser.error("pass --disease <name> or --all")
    for d in (config.diseases() if args.all else [args.disease]):
        run_training(d)


if __name__ == "__main__":
    main()
