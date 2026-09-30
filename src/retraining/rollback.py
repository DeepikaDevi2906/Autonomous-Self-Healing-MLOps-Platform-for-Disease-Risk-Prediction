"""
Reverts to the previous champion.

    python src/retraining/rollback.py --disease diabetes

Calling it twice goes two steps back (it no longer bounces between the last two
versions). The drift reference is not changed automatically; run the next
monitoring check to confirm the restored model is healthy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config
from monitoring.alerting import send_alert
from training import model_registry


def rollback_to_previous(disease: str) -> str | None:
    current = model_registry.get_champion_version(disease)
    restored = model_registry.rollback(disease)
    if restored:
        send_alert(disease, "ROLLBACK", {
            "from_version": current.version if current else None, "to_version": restored,
        })
    return restored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", choices=config.diseases(), required=True)
    rollback_to_previous(parser.parse_args().disease)
