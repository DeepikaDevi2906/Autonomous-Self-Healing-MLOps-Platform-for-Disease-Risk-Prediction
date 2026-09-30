"""
Runs drift + performance checks for one disease on its newest unchecked batch.

Each batch is checked once: the last checked batch is remembered in
data/monitoring/state.json, so a daily schedule no longer re-checks batch_1
forever. Every result is appended to data/monitoring/monitoring_history.jsonl.

    python src/monitoring/monitor.py --disease diabetes
    python src/monitoring/monitor.py --all --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from common import config
from common.storage import append_jsonl, read_json, read_jsonl, utc_now, write_json
from data_pipeline.batch_loader import batch_number, latest_batch, list_batches, load_batch
from monitoring.alerting import send_alert
from monitoring.drift_detector import detect_drift
from monitoring.performance_monitor import check_performance


def history_path():
    return config.monitoring_dir() / "monitoring_history.jsonl"


def state_path():
    return config.monitoring_dir() / "state.json"


def last_checked(disease: str) -> str | None:
    return read_json(state_path(), {}).get(disease, {}).get("last_checked_batch")


def mark_checked(disease: str, batch_name: str) -> None:
    state = read_json(state_path(), {})
    state.setdefault(disease, {})["last_checked_batch"] = batch_name
    state[disease]["checked_at"] = utc_now()
    write_json(state_path(), state)


def unchecked_batches(disease: str) -> list[str]:
    done = last_checked(disease)
    batches = list_batches(disease)
    if done is None:
        return batches
    return [b for b in batches if batch_number(b) > batch_number(done)]


def cooldown_remaining(disease: str, batch_name: str) -> int:
    """Batches still to wait before another automatic retrain (0 = allowed)."""
    cooldown = config.settings()["retraining"].get("cooldown_batches", 0)
    runs = [r for r in read_jsonl(config.monitoring_dir() / "retraining_history.jsonl")
            if r["disease"] == disease and r.get("batches_used")]
    if not cooldown or not runs:
        return 0
    last_used = max(batch_number(b) for b in runs[-1]["batches_used"])
    return max(cooldown - (batch_number(batch_name) - last_used), 0)


def run_monitoring(disease: str, batch_name: str | None = None, force: bool = False) -> dict:
    """
    batch_name=None -> newest unchecked batch (or newest batch when force=True).
    Returns a dict with retraining_needed and the reasons.
    """
    if batch_name is None:
        pending = unchecked_batches(disease)
        batch_name = pending[-1] if pending else (latest_batch(disease) if force else None)

    if batch_name is None:
        print(f"{disease}: no new batch to check.")
        return {"disease": disease, "status": "no_new_batch", "retraining_needed": False, "reasons": []}

    print(f"\n===== Monitoring {disease} ({batch_name}) =====")
    batch = load_batch(disease, batch_name)
    min_rows = config.settings()["monitoring"]["min_batch_rows"]
    if len(batch) < min_rows:
        print(f"  Skipping: batch has {len(batch)} rows (< {min_rows})")
        mark_checked(disease, batch_name)
        return {"disease": disease, "batch_name": batch_name, "status": "too_small",
                "retraining_needed": False, "reasons": []}

    drift = detect_drift(disease, batch_name, batch)
    labelled = "Outcome" in batch.columns and batch["Outcome"].notna().any()
    if labelled:
        performance = check_performance(disease, batch_name, batch)
    else:
        # uploaded without outcomes: accuracy cannot be measured, only drift
        from training import model_registry

        champion = model_registry.get_champion_version(disease)
        if champion is None:
            raise LookupError(f"No champion model registered for {disease}. Run initial training first.")
        performance = {"model_version": str(champion.version), "batch_auc": None, "batch_auc_ci": [None, None],
                       "baseline_auc": float(champion.tags["test_auc"]) if champion.tags.get("test_auc") else None,
                       "auc_drop": None, "performance_dropped": False}
        print("  No Outcome column: drift check only")

    reasons = []
    if drift["drift_detected"]:
        reasons.append("DATA_DRIFT")
        send_alert(disease, "DATA_DRIFT", {k: v for k, v in drift.items() if k != "columns"})
    if performance["performance_dropped"]:
        reasons.append("PERFORMANCE_DROP")
        send_alert(disease, "PERFORMANCE_DROP", performance)

    waiting = cooldown_remaining(disease, batch_name) if reasons else 0
    note = None
    if reasons and not labelled:
        # nothing new to learn from without outcomes: alert, but do not retrain
        note = "Drift found in a batch without outcomes. Upload the outcomes to allow retraining."
        print(f"  {note}")
    if waiting:
        print(f"  Retraining on cooldown: {waiting} more batch(es) needed since the last retrain")

    record = {
        "timestamp": utc_now(),
        "disease": disease,
        "batch_name": batch_name,
        "status": "checked",
        "rows": len(batch),
        "labelled": bool(labelled),
        "model_version": performance["model_version"],
        "drift_share": drift["drift_share"],
        "drift_detected": drift["drift_detected"],
        "drifted_columns": [c for c, v in drift["columns"].items() if v["drifted"]],
        "batch_auc": performance["batch_auc"],
        "batch_auc_ci": performance["batch_auc_ci"],
        "baseline_auc": performance["baseline_auc"],
        "auc_drop": performance["auc_drop"],
        "performance_dropped": performance["performance_dropped"],
        "retraining_needed": bool(reasons) and not waiting and bool(labelled),
        "note": note,
        "cooldown_remaining": waiting,
        "reasons": reasons,
    }
    append_jsonl(history_path(), record)
    mark_checked(disease, batch_name)

    if record["retraining_needed"]:
        print(f"{disease}: retraining needed {reasons}")
    elif reasons:
        print(f"{disease}: issues {reasons} but retraining is on cooldown")
    else:
        print(f"{disease}: healthy, no retraining needed")
    return record


def monitoring_history(disease: str | None = None, limit: int = 200) -> list[dict]:
    rows = read_jsonl(history_path())
    if disease:
        rows = [r for r in rows if r["disease"] == disease]
    return rows[-limit:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", choices=config.diseases())
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--batch", default=None)
    parser.add_argument("--force", action="store_true", help="re-check the newest batch")
    args = parser.parse_args()
    if not args.all and not args.disease:
        parser.error("pass --disease <name> or --all")
    for d in (config.diseases() if args.all else [args.disease]):
        run_monitoring(d, args.batch, args.force)


if __name__ == "__main__":
    main()
