"""
Background jobs (setup, retraining) run one at a time on a single worker thread.
That serialises every write to the MLflow registry, which also avoids the old
"database is locked" problem when three retrains started together.
"""

from __future__ import annotations

import io
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from typing import Callable

from common.storage import to_jsonable, utc_now

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlops-job")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
MAX_LOG_CHARS = 20_000


class _Tee(io.TextIOBase):
    def __init__(self, job: dict):
        self.job = job

    def write(self, s: str) -> int:
        self.job["log"] = (self.job["log"] + s)[-MAX_LOG_CHARS:]
        return len(s)


def submit(kind: str, target: str | None, fn: Callable, *args, on_done: Callable | None = None) -> dict:
    with _lock:
        for job in _jobs.values():
            if job["kind"] == kind and job["target"] == target and job["status"] in ("queued", "running"):
                raise RuntimeError(f"A {kind} job for {target or 'all'} is already {job['status']}")
        job = {"id": uuid.uuid4().hex[:12], "kind": kind, "target": target, "status": "queued",
               "created_at": utc_now(), "started_at": None, "finished_at": None,
               "result": None, "error": None, "log": ""}
        _jobs[job["id"]] = job

    def run():
        job["status"], job["started_at"] = "running", utc_now()
        try:
            with redirect_stdout(_Tee(job)):
                job["result"] = to_jsonable(fn(*args))
            job["status"] = "succeeded"
            if on_done:
                on_done()
        except Exception as exc:
            job["status"], job["error"] = "failed", f"{exc}\n{traceback.format_exc()[-3000:]}"
        finally:
            job["finished_at"] = utc_now()

    _executor.submit(run)
    return public(job)


def public(job: dict) -> dict:
    return {k: v for k, v in job.items()}


def get(job_id: str) -> dict | None:
    job = _jobs.get(job_id)
    return public(job) if job else None


def list_jobs(limit: int = 20) -> list[dict]:
    jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)[:limit]
    return [{k: v for k, v in j.items() if k != "log"} for j in jobs]


def busy(kind: str | None = None) -> bool:
    return any(j["status"] in ("queued", "running") and (kind is None or j["kind"] == kind)
               for j in _jobs.values())
