"""Picks the model with the highest VALIDATION score (the test set is never used to choose)."""

from __future__ import annotations


def select_champion(model_scores: dict[str, float | None]) -> str:
    """
    model_scores: {"RandomForest": 0.84, "XGBoost": 0.82, "NeuralNet": None}
    Models without a score are skipped. Ties go to the first model listed.
    """
    valid = {name: s for name, s in model_scores.items() if s is not None}
    if not valid:
        raise ValueError("No model produced a valid score")

    best_name = max(valid, key=lambda n: (valid[n], -list(valid).index(n)))
    for name, score in model_scores.items():
        print(f"  {name}: {score}")
    print(f"Champion: {best_name} (score = {valid[best_name]})")
    return best_name
