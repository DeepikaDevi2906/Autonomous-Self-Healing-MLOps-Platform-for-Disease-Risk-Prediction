"""Model scoring. AUC-ROC is the selection metric; the others are reported."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def auc(model, X, y) -> float | None:
    """AUC-ROC, or None when y has a single class (AUC is undefined)."""
    if y is None or len(set(y)) < 2:
        return None
    return round(float(roc_auc_score(y, model.predict_proba(X)[:, 1])), 4)


def evaluate(model, X, y) -> dict:
    preds = model.predict(X)
    return {
        "auc": auc(model, X, y),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
    }
