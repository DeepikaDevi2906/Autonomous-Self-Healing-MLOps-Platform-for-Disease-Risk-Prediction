"""
In-memory cache of (model, preprocessor, version) per disease.

Models used to be loaded from MLflow on EVERY request. Now they are loaded once
and refreshed only when the champion alias points at a different version
(checked at most every REFRESH_SECONDS, or immediately after promote/rollback).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from training import model_registry

REFRESH_SECONDS = 30


class ModelNotAvailable(RuntimeError):
    pass


@dataclass
class LoadedModel:
    model: object
    preprocessor: object
    version: str
    algorithm: str
    tags: dict
    loaded_at: float


class ModelStore:
    def __init__(self):
        self._models: dict[str, LoadedModel] = {}
        self._checked: dict[str, float] = {}
        self._lock = threading.Lock()

    def _load(self, disease: str) -> LoadedModel:
        try:
            model, version = model_registry.load_champion(disease)
            preprocessor = model_registry.load_preprocessor(version)
        except LookupError as exc:
            raise ModelNotAvailable(str(exc)) from exc
        except Exception as exc:
            raise ModelNotAvailable(f"Could not load the {disease} champion: {exc}") from exc
        return LoadedModel(model, preprocessor, str(version.version),
                           version.tags.get("algorithm", "unknown"), dict(version.tags), time.time())

    def get(self, disease: str) -> LoadedModel:
        with self._lock:
            cached = self._models.get(disease)
            now = time.time()
            if cached and now - self._checked.get(disease, 0) < REFRESH_SECONDS:
                return cached
            try:
                current = model_registry.get_champion_version(disease)
            except Exception:
                current = None  # registry briefly unreachable: keep serving the cached model
                if cached:
                    return cached
            self._checked[disease] = now
            if cached and current is not None and str(current.version) == cached.version:
                return cached
            self._models[disease] = self._load(disease)
            return self._models[disease]

    def invalidate(self, disease: str | None = None) -> None:
        with self._lock:
            if disease:
                self._models.pop(disease, None)
                self._checked.pop(disease, None)
            else:
                self._models.clear()
                self._checked.clear()

    def loaded(self) -> dict[str, str]:
        return {d: m.version for d, m in self._models.items()}


store = ModelStore()
