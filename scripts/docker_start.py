"""
Container entrypoint for the API service.

1. Waits until the MLflow tracking server answers.
2. On first start (no champion models yet) runs the full setup automatically:
   validate -> preprocess -> train all diseases -> first batch.
3. Starts the API with a single uvicorn worker.

Written in Python rather than shell, so Windows line endings can never break it.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def wait_for_mlflow(timeout: int = 180) -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("http"):
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=5)
            print(f"MLflow is ready at {uri}", flush=True)
            return
        except Exception:
            print("Waiting for MLflow...", flush=True)
            time.sleep(3)
    raise SystemExit(f"MLflow at {uri} did not become ready within {timeout}s")


def needs_setup() -> bool:
    from common import config
    from training import model_registry

    return any(model_registry.get_champion_version(d) is None for d in config.diseases())


if __name__ == "__main__":
    wait_for_mlflow()
    if os.environ.get("AUTO_SETUP", "true").lower() == "true" and needs_setup():
        print("First start: training all models (about a minute)...", flush=True)
        from pipeline import setup

        setup()
        print("Setup complete.", flush=True)
    else:
        print("Models already trained; skipping setup.", flush=True)

    os.chdir(ROOT / "src")
    os.execvp("uvicorn", ["uvicorn", "inference.main:app", "--host", "0.0.0.0",
                          "--port", "8000", "--workers", "1"])
