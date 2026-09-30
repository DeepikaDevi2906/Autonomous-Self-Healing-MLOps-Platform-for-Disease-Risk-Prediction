"""
One-command setup: validate -> preprocess -> train all diseases -> first clean batch.

    python scripts/bootstrap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import setup  # noqa: E402

if __name__ == "__main__":
    result = setup()
    print("\nSetup complete:")
    for disease, info in result["trained"].items():
        print(f"  {disease:14s} {info['champion_algorithm']:12s} v{info['version']}  test AUC {info['test_auc']}")
