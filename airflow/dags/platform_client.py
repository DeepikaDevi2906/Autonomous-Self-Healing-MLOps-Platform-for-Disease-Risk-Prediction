"""
Tiny HTTP client used by the DAGs. Airflow only orchestrates; the API service
does the ML work. That keeps the Airflow image free of mlflow/evidently/xgboost,
so their dependencies can never clash with Airflow's own.

PLATFORM_API_URL defaults to http://api:8000 (the docker-compose service name).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

API_URL = os.environ.get("PLATFORM_API_URL", "http://api:8000").rstrip("/")
DISEASES = ["diabetes", "heart_disease", "breast_cancer"]


def call(method: str, path: str, body: dict | None = None, timeout: int = 600) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{API_URL}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} failed with {exc.code}: {exc.read().decode()[:500]}") from exc


def wait_for_job(job: dict, timeout_seconds: int = 3600, poll_seconds: int = 10) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        current = call("GET", f"/api/jobs/{job['id']}")
        if current["status"] == "succeeded":
            print(current.get("log", "")[-4000:])
            return current["result"]
        if current["status"] == "failed":
            raise RuntimeError(f"Job {job['id']} failed:\n{current.get('error')}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Job {job['id']} did not finish within {timeout_seconds}s")
