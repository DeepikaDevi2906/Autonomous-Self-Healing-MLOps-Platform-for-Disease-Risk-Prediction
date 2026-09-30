"""
Alerts are printed, appended to data/monitoring/alerts.jsonl and, if the
ALERT_WEBHOOK_URL environment variable is set (Slack/Teams/any JSON webhook),
posted there too.
"""

from __future__ import annotations

import json
import os
import urllib.request

from common import config
from common.storage import append_jsonl, read_jsonl, utc_now

SEVERITY = {
    "DATA_DRIFT": "warning",
    "PERFORMANCE_DROP": "critical",
    "MODEL_PROMOTED": "info",
    "CHALLENGER_REJECTED": "info",
    "ROLLBACK": "warning",
    "MONITORING_ERROR": "critical",
}


def alerts_path():
    return config.monitoring_dir() / "alerts.jsonl"


def _post_webhook(alert: dict) -> None:
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        return
    body = json.dumps({"text": f"[{alert['severity']}] {alert['disease']}: {alert['reason']}", **alert})
    request = urllib.request.Request(url, data=body.encode(), headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=5)
    except Exception as exc:  # an alert channel outage must never break the pipeline
        print(f"  Could not deliver webhook alert: {exc}")


def send_alert(disease: str, reason: str, details: dict) -> dict:
    alert = {
        "timestamp": utc_now(),
        "disease": disease,
        "reason": reason,
        "severity": SEVERITY.get(reason, "info"),
        "details": details,
    }
    print(f"\nALERT [{alert['severity']}] {disease}: {reason}")
    for key, value in details.items():
        print(f"  {key}: {value}")
    alert = append_jsonl(alerts_path(), alert)
    _post_webhook(alert)
    return alert


def recent_alerts(limit: int = 50, disease: str | None = None) -> list[dict]:
    rows = read_jsonl(alerts_path())
    if disease:
        rows = [r for r in rows if r["disease"] == disease]
    return list(reversed(rows[-limit:]))
