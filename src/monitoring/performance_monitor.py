"""
Performance check: how does the CURRENT champion score on a labelled batch?

The baseline is the champion's own test AUC, read from its registry tags, so it
is always correct after a retrain or rollback (it used to be hard-coded).

Batches are small, so a single AUC number is noisy. A drop only counts when it
is larger than the threshold AND it is statistically significant: the 5th
percentile of the bootstrapped (test AUC - batch AUC) difference is above zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import config
from training import model_registry
from training.evaluation import evaluate


def _bootstrap_aucs(y, probs, n_boot: int, seed: int) -> np.ndarray:
    y, probs = np.asarray(y), np.asarray(probs)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            scores.append(roc_auc_score(y[idx], probs[idx]))
    return np.array(scores)


def drop_is_significant(model, X_test, y_test, X_batch, y_batch, n_boot: int = 500):
    """
    Bootstraps the champion's AUC on the test set AND on the batch.
    Returns (batch_ci_low, batch_ci_high, drop_ci_low): the drop is significant
    when drop_ci_low (5th percentile of test_auc - batch_auc) is above zero.
    """
    batch = _bootstrap_aucs(y_batch, model.predict_proba(X_batch)[:, 1], n_boot, seed=1)
    test = _bootstrap_aucs(y_test, model.predict_proba(X_test)[:, 1], n_boot, seed=2)
    if len(batch) < n_boot / 2 or len(test) < n_boot / 2:
        return None, None, None
    n = min(len(batch), len(test))
    diff = test[:n] - batch[:n]
    return (round(float(np.percentile(batch, 5)), 4), round(float(np.percentile(batch, 95)), 4),
            round(float(np.percentile(diff, 5)), 4))


def check_performance(disease: str, batch_name: str, batch_data: pd.DataFrame | None = None) -> dict:
    threshold = config.settings()["monitoring"]["performance_drop_threshold"]
    if batch_data is None:
        batch_data = pd.read_csv(config.batches_dir(disease) / batch_name / "processed_data.csv")
    model, version = model_registry.load_champion(disease)
    if "Outcome" not in batch_data.columns:
        print(f"  {batch_name} has no Outcome column: drift check only, no performance check")
        baseline = version.tags.get("test_auc")
        return {"disease": disease, "batch_name": batch_name, "model_version": str(version.version),
                "baseline_auc": float(baseline) if baseline not in (None, "None") else None,
                "batch_auc": None, "batch_auc_ci": [None, None], "drop_ci_low": None,
                "batch_accuracy": None, "batch_f1": None, "auc_drop": None,
                "threshold": threshold, "performance_dropped": False, "labelled": False}

    X, y = batch_data[config.features(disease)], batch_data["Outcome"]
    metrics = evaluate(model, X, y)

    baseline = version.tags.get("test_auc")
    baseline_auc = float(baseline) if baseline not in (None, "None") else None
    batch_auc = metrics["auc"]

    ci_low = ci_high = drop_ci_low = None
    if batch_auc is not None:
        test = pd.read_csv(config.processed_dir(disease) / "test.csv")
        ci_low, ci_high, drop_ci_low = drop_is_significant(
            model, test[config.features(disease)], test["Outcome"], X, y)

    if batch_auc is None or baseline_auc is None:
        auc_drop, dropped = None, False
        print(f"  AUC cannot be compared for {batch_name} (single-class batch or missing baseline)")
    else:
        auc_drop = round(baseline_auc - batch_auc, 4)
        significant = drop_ci_low is None or drop_ci_low > 0
        dropped = auc_drop > threshold and significant

    print(f"Performance {disease}/{batch_name}: baseline {baseline_auc}, batch {batch_auc}, "
          f"(90% CI {ci_low}-{ci_high}), drop {auc_drop} -> retrain={dropped}")
    return {
        "disease": disease,
        "batch_name": batch_name,
        "model_version": str(version.version),
        "baseline_auc": baseline_auc,
        "batch_auc": batch_auc,
        "batch_auc_ci": [ci_low, ci_high],
        "drop_ci_low": drop_ci_low,
        "batch_accuracy": metrics["accuracy"],
        "batch_f1": metrics["f1"],
        "auc_drop": auc_drop,
        "threshold": threshold,
        "performance_dropped": dropped,
    }
