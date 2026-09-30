"""
Single source of truth for paths, settings and MLflow configuration.

Every path is resolved from the project root, never from the current working
directory, so code behaves the same from a terminal, pytest, Docker or Airflow.

Environment variables:
    MLOPS_PROJECT_ROOT   project root (default: two folders above this file)
    MLFLOW_TRACKING_URI  MLflow server/database (default: local SQLite file)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml


def project_root() -> Path:
    env = os.environ.get("MLOPS_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def path(*parts: str) -> Path:
    """Absolute path inside the project, e.g. path("data", "raw")."""
    return project_root().joinpath(*parts)


@lru_cache(maxsize=4)
def _load_settings(settings_file: str) -> dict:
    with open(settings_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def settings() -> dict:
    return _load_settings(str(path("config", "settings.yaml")))


def diseases() -> list[str]:
    return list(settings()["diseases"].keys())


def disease_settings(disease: str) -> dict:
    all_diseases = settings()["diseases"]
    if disease not in all_diseases:
        raise ValueError(f"Unknown disease '{disease}'. Expected one of {list(all_diseases)}")
    return all_diseases[disease]


def features(disease: str) -> list[str]:
    return list(disease_settings(disease)["features"])


# ── Standard locations ────────────────────────────────────────────────

def processed_dir(disease: str) -> Path:
    return path("data", "processed", disease)


def reference_path(disease: str) -> Path:
    return path("data", "reference", disease, "reference_data.csv")


def batches_dir(disease: str) -> Path:
    return path("data", "batches", disease)


def monitoring_dir() -> Path:
    return path("data", "monitoring")


def preprocessor_path(disease: str) -> Path:
    return processed_dir(disease) / "preprocessor.pkl"


# ── MLflow ────────────────────────────────────────────────────────────

def mlflow_tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    db = path("mlflow", "mlflow.db")
    db.parent.mkdir(parents=True, exist_ok=True)
    # as_posix() keeps Windows paths valid; an absolute path gives sqlite:////...
    return f"sqlite:///{db.as_posix()}"


def uses_local_store() -> bool:
    return not mlflow_tracking_uri().startswith(("http://", "https://", "databricks"))


def configure_mlflow() -> None:
    import mlflow

    mlflow.set_tracking_uri(mlflow_tracking_uri())


def set_experiment(name: str) -> None:
    """
    Creates the experiment if needed. With a local store the artifact location
    is pinned inside the project (mlflow/artifacts/<name>) instead of MLflow's
    default "./mlruns relative to wherever you happened to run the command".
    With a tracking server, the server decides (proxied artifacts).
    """
    import mlflow

    configure_mlflow()
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        artifact_location = None
        if uses_local_store():
            artifact_dir = path("mlflow", "artifacts", name)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_location = artifact_dir.as_uri()
        client.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)


def registered_model_name(disease: str) -> str:
    return f"{disease}{settings()['mlflow']['model_name_suffix']}"


def champion_alias() -> str:
    return settings()["mlflow"]["champion_alias"]
