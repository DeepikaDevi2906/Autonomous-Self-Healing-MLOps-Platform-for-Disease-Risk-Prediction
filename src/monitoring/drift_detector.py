"""
Data drift: does a new batch look different from the champion's training data?

Evidently runs a statistical test per column (K-S for small numeric samples).
The batch counts as drifted when more than `drift_share_threshold` of columns
drifted. An HTML report is saved next to the batch.
"""

from __future__ import annotations

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from common import config


def _parse_snapshot(snapshot_dict: dict) -> tuple[dict, dict]:
    summary, columns = None, {}
    for metric in snapshot_dict["metrics"]:
        cfg = metric.get("config", {})
        metric_type = cfg.get("type", "")
        if metric_type.endswith("DriftedColumnsCount"):
            summary = metric["value"]
        elif metric_type.endswith("ValueDrift"):
            method = cfg.get("method", "")
            value, threshold = metric["value"], cfg.get("threshold")
            # p-value tests drift when value < threshold; distance tests when value >= threshold
            is_p_value = "p_value" in method
            drifted = value < threshold if is_p_value else value >= threshold
            columns[cfg["column"]] = {
                "method": method, "score": round(float(value), 6), "drifted": bool(drifted)
            }
    if summary is None:
        raise RuntimeError("Evidently output has no DriftedColumnsCount metric; check the evidently version")
    return summary, columns


def detect_drift(disease: str, batch_name: str, batch_data: pd.DataFrame | None = None) -> dict:
    threshold = config.settings()["monitoring"]["drift_share_threshold"]
    reference = pd.read_csv(config.reference_path(disease))

    if batch_data is None:
        batch_data = pd.read_csv(config.batches_dir(disease) / batch_name / "processed_data.csv")
    current = batch_data[reference.columns]

    print(f"Checking drift for {disease} / {batch_name} "
          f"(reference {len(reference)} rows, batch {len(current)} rows)")

    snapshot = Report([DataDriftPreset(drift_share=threshold)]).run(
        reference_data=reference, current_data=current
    )
    summary, columns = _parse_snapshot(snapshot.dict())

    report_file = config.batches_dir(disease) / batch_name / "drift_report.html"
    try:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(report_file))
    except Exception as exc:
        print(f"  Could not save drift HTML report: {exc}")

    drift_share = round(float(summary["share"]), 4)
    result = {
        "disease": disease,
        "batch_name": batch_name,
        "drift_share": drift_share,
        "drifted_count": int(summary["count"]),
        "total_columns": len(reference.columns),
        "threshold": threshold,
        "drift_detected": drift_share > threshold,
        "columns": columns,
    }
    print(f"  Drifted columns: {result['drifted_count']}/{result['total_columns']} "
          f"(share {drift_share}, threshold {threshold}) -> drift={result['drift_detected']}")
    return result
