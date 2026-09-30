"""
Logs every prediction (raw inputs + result) to
data/monitoring/predictions/<disease>.csv, one file per disease because every
disease has different columns. (Replaces middleware.py, which was never wired
in, only knew diabetes fields, and read attributes the response did not have.)
"""

from __future__ import annotations

import csv
import threading

from common import config
from common.storage import utc_now

_lock = threading.Lock()
META = ["timestamp", "prediction", "probability", "model_version", "model_algorithm"]


def log_path(disease: str):
    return config.monitoring_dir() / "predictions" / f"{disease}.csv"


def log_prediction(disease: str, inputs: dict, result: dict) -> None:
    fields = ["timestamp"] + config.features(disease) + META[1:]
    row = {"timestamp": utc_now(), **inputs, **{k: result[k] for k in META[1:]}}
    file = log_path(disease)
    with _lock:
        file.parent.mkdir(parents=True, exist_ok=True)
        new = not file.exists()
        with open(file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow(row)


def recent_predictions(disease: str, limit: int = 50) -> list[dict]:
    file = log_path(disease)
    if not file.exists():
        return []
    with _lock, open(file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return list(reversed(rows[-limit:]))


def prediction_count(disease: str) -> int:
    file = log_path(disease)
    if not file.exists():
        return 0
    with _lock, open(file, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)
