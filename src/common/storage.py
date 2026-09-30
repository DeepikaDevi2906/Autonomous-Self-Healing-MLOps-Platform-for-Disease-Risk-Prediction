"""Small, process-safe helpers for the JSON-lines logs used by monitoring."""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_jsonable(value: Any) -> Any:
    """Converts numpy / pandas scalars so json.dumps never fails."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def append_jsonl(file_path: Path, record: dict) -> dict:
    record = to_jsonable(record)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_jsonl(file_path: Path, limit: int | None = None) -> list[dict]:
    if not file_path.exists():
        return []
    with _lock, open(file_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[-limit:] if limit else rows


def read_json(file_path: Path, default: Any = None) -> Any:
    if not file_path.exists():
        return default
    with _lock, open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = file_path.with_suffix(file_path.suffix + ".tmp")
    with _lock:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(data), f, indent=2)
        tmp.replace(file_path)
