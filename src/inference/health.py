"""/health - one entry per disease, based on the real per-disease champions."""

from __future__ import annotations

from datetime import datetime, timezone

from common import config
from training import model_registry

_start_time = datetime.now(timezone.utc)


def get_health_status() -> dict:
    models, registry_ok = {}, True
    for disease in config.diseases():
        try:
            version = model_registry.get_champion_version(disease)
        except Exception as exc:
            registry_ok = False
            models[disease] = {"status": "unknown", "error": str(exc)}
            continue
        if version is None:
            models[disease] = {"status": "missing"}
        else:
            models[disease] = {
                "status": "ready",
                "version": str(version.version),
                "algorithm": version.tags.get("algorithm", "unknown"),
                "test_auc": version.tags.get("test_auc"),
            }

    ready = sum(m["status"] == "ready" for m in models.values())
    status = "healthy" if ready == len(models) else ("degraded" if ready else "unavailable")
    now = datetime.now(timezone.utc)
    return {
        "status": status,
        "registry_reachable": registry_ok,
        "tracking_uri": config.mlflow_tracking_uri() if not config.uses_local_store() else "local sqlite",
        "models": models,
        "uptime_seconds": int((now - _start_time).total_seconds()),
        "timestamp": now.isoformat(timespec="seconds"),
    }
