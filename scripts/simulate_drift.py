"""
Demonstrates the self-healing loop for one disease:
create a drifted batch -> monitor it -> retrain if needed -> print the decision.

    python scripts/simulate_drift.py --disease diabetes --drift 1.5
    python scripts/simulate_drift.py --disease heart_disease --concept-shift 0.8
    python scripts/simulate_drift.py --disease heart_disease --label-noise 0.3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import config  # noqa: E402
from data_pipeline.create_batches import create_batch  # noqa: E402
from pipeline import monitor_and_heal  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", choices=config.diseases(), default="diabetes")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--drift", type=float, default=1.5)
    parser.add_argument("--label-noise", type=float, default=0.0)
    parser.add_argument("--concept-shift", type=float, default=0.0)
    args = parser.parse_args()

    create_batch(args.disease, args.rows, args.drift, args.label_noise, None, args.concept_shift)
    result = monitor_and_heal(args.disease)
    m, r = result["monitoring"], result["retraining"]
    print("\nMonitoring:", json.dumps({k: m.get(k) for k in ("batch_name", "drift_share", "batch_auc", "reasons")}))
    if r:
        print("Retraining:", json.dumps({k: r.get(k) for k in ("challenger_version", "should_promote", "decision")}))
