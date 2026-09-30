"""Promotes a registered challenger and moves the drift baseline with it."""

from __future__ import annotations

from monitoring.alerting import send_alert
from training import model_registry


def promote_challenger(disease: str, version: str, algorithm: str, processed_train, comparison: dict) -> str:
    model_registry.promote(disease, version, reason=comparison.get("decision", "retraining"))
    model_registry.save_reference_from(disease, processed_train)
    send_alert(disease, "MODEL_PROMOTED", {
        "version": version, "algorithm": algorithm,
        "previous_version": comparison.get("champion_version"),
        "decision": comparison.get("decision"),
    })
    return version
