"""
Champion vs challenger.

Both models are scored on:
  * recent data - the held-out part of the newest batch (what the model sees NOW)
  * the original test set - guards against forgetting the historical population

The challenger is promoted when it beats the champion on recent data by at least
`min_improvement` and does not lose more than `max_test_regression` on the test set.
If the champion has degraded (the reason retraining was triggered), any real gain
on recent data is enough. With no batches yet, the test set decides.
"""

from __future__ import annotations

import pandas as pd

from common import config
from training import model_registry
from training.evaluation import auc


def _score(model, df: pd.DataFrame | None, features: list[str]) -> float | None:
    if df is None or df.empty:
        return None
    return auc(model, df[features], df["Outcome"])


def compare_models(
    disease: str, challenger, test_df: pd.DataFrame, recent_eval: pd.DataFrame | None,
    trigger_reasons: list[str] | None = None,
) -> dict:
    rules = config.settings()["retraining"]
    feats = config.features(disease)
    trigger_reasons = trigger_reasons or []

    try:
        champion, champion_version = model_registry.load_champion(disease)
    except LookupError:
        return {"disease": disease, "should_promote": True, "decision": "no champion exists yet",
                "champion_version": None, "challenger_test_auc": _score(challenger, test_df, feats)}

    result = {
        "disease": disease,
        "champion_version": str(champion_version.version),
        "champion_test_auc": _score(champion, test_df, feats),
        "challenger_test_auc": _score(challenger, test_df, feats),
        "champion_recent_auc": _score(champion, recent_eval, feats),
        "challenger_recent_auc": _score(challenger, recent_eval, feats),
        "trigger_reasons": trigger_reasons,
    }
    test_delta = round(result["challenger_test_auc"] - result["champion_test_auc"], 4)
    result["test_delta"] = test_delta
    keeps_history = test_delta >= -rules["max_test_regression"]

    if result["champion_recent_auc"] is None or result["challenger_recent_auc"] is None:
        result["recent_delta"] = None
        promote = test_delta >= rules["min_improvement"]
        decision = (f"no recent data; test AUC change {test_delta:+.4f} "
                    f"(needs >= {rules['min_improvement']})")
    else:
        recent_delta = round(result["challenger_recent_auc"] - result["champion_recent_auc"], 4)
        result["recent_delta"] = recent_delta
        degraded = "PERFORMANCE_DROP" in trigger_reasons
        needed = 0.0 if degraded else rules["min_improvement"]
        gain_ok = recent_delta > needed if degraded else recent_delta >= needed
        promote = gain_ok and keeps_history
        decision = (f"recent AUC change {recent_delta:+.4f} (needs {'>' if degraded else '>='} {needed}), "
                    f"test AUC change {test_delta:+.4f} (limit -{rules['max_test_regression']})")

    result["should_promote"] = bool(promote)
    result["decision"] = decision
    print(f"\nComparing models for {disease}: {decision} -> promote={promote}")
    return result
