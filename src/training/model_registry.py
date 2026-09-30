"""
Everything that touches the MLflow Model Registry lives here.

Uses the "champion" alias instead of the deprecated Production/Archived stages.
Promotion history is stored as a registered-model tag, so rollback always goes
one step further back instead of bouncing between the last two versions.

Each registered model version logs the fitted DiseasePreprocessor as an artifact
in the same run, so a model can never be served with a mismatched scaler.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from common import config
from common.storage import utc_now

HISTORY_TAG = "champion_history"


def client() -> mlflow.tracking.MlflowClient:
    config.configure_mlflow()
    return mlflow.tracking.MlflowClient()


def get_champion_version(disease: str):
    try:
        return client().get_model_version_by_alias(
            config.registered_model_name(disease), config.champion_alias()
        )
    except MlflowException:
        return None


def load_champion(disease: str):
    """Returns (model, model_version) or raises LookupError when there is no champion."""
    version = get_champion_version(disease)
    if version is None:
        raise LookupError(f"No champion model registered for {disease}. Run initial training first.")
    model = mlflow.sklearn.load_model(f"models:/{version.name}/{version.version}")
    return model, version


def load_preprocessor(version):
    local = mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="preprocessor/preprocessor.pkl",
        tracking_uri=config.mlflow_tracking_uri(),
    )
    return joblib.load(local)


def register_model(disease: str, model, algorithm: str, X_sample, metrics: dict, tags: dict | None = None):
    """Logs model + preprocessor in the ACTIVE run and registers a new version (not yet champion)."""
    name = config.registered_model_name(disease)
    signature = infer_signature(X_sample, model.predict_proba(X_sample))
    info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=name,
        signature=signature,
        serialization_format="cloudpickle",
    )
    mlflow.log_artifact(str(config.preprocessor_path(disease)), artifact_path="preprocessor")
    version = str(info.registered_model_version)

    c = client()
    all_tags = {"algorithm": algorithm, "registered_at": utc_now(), **(tags or {})}
    for key, value in metrics.items():
        if value is not None:
            all_tags[key] = str(value)
    for key, value in all_tags.items():
        c.set_model_version_tag(name, version, key, str(value))
    return version


def champion_history(disease: str) -> list[str]:
    try:
        model = client().get_registered_model(config.registered_model_name(disease))
    except MlflowException:
        return []
    return json.loads(model.tags.get(HISTORY_TAG, "[]"))


def _save_history(disease: str, history: list[str]) -> None:
    # Keep the tag small; 50 promotions of history is plenty for rollback.
    client().set_registered_model_tag(
        config.registered_model_name(disease), HISTORY_TAG, json.dumps(history[-50:])
    )


def promote(disease: str, version: str, reason: str) -> str:
    name = config.registered_model_name(disease)
    c = client()
    c.set_registered_model_alias(name, config.champion_alias(), str(version))
    c.set_model_version_tag(name, str(version), "promoted_at", utc_now())
    c.set_model_version_tag(name, str(version), "promotion_reason", reason)

    history = champion_history(disease)
    if not history or history[-1] != str(version):
        history.append(str(version))
    _save_history(disease, history)
    print(f"  Version {version} is now the champion for {disease}")
    return str(version)


def rollback(disease: str) -> str | None:
    """Makes the previous champion live again. Returns its version, or None if impossible."""
    history = champion_history(disease)
    if len(history) < 2:
        print(f"  No earlier champion for {disease}; cannot roll back.")
        return None

    name = config.registered_model_name(disease)
    c = client()
    bad = history.pop()
    previous = history[-1]
    c.set_registered_model_alias(name, config.champion_alias(), previous)
    c.set_model_version_tag(name, bad, "rolled_back_at", utc_now())
    c.set_model_version_tag(name, previous, "restored_at", utc_now())
    _save_history(disease, history)
    print(f"  Rolled back {disease}: version {bad} -> {previous}")
    return previous


def list_versions(disease: str) -> list[dict]:
    name = config.registered_model_name(disease)
    try:
        versions = client().search_model_versions(f"name='{name}'")
    except MlflowException:
        return []
    champion = get_champion_version(disease)
    champion_v = champion.version if champion else None
    rows = []
    for v in versions:
        rows.append({
            "version": str(v.version),
            "is_champion": str(v.version) == str(champion_v),
            "run_id": v.run_id,
            "created_at": v.creation_timestamp,
            "tags": dict(v.tags),
        })
    return sorted(rows, key=lambda r: int(r["version"]), reverse=True)


def save_reference_from(disease: str, processed_train):
    """The drift baseline follows the champion: after a promotion it becomes that model's training data."""
    ref = config.reference_path(disease)
    ref.parent.mkdir(parents=True, exist_ok=True)
    processed_train.drop(columns=["Outcome"], errors="ignore").to_csv(ref, index=False)
    return ref
