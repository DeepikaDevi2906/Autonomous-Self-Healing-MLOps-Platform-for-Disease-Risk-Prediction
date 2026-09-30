"""
FastAPI service: predictions + the operations API used by the React dashboard
and by Airflow.

    uvicorn inference.main:app --port 8000        (from the src/ folder)
    http://localhost:8000/docs

Run with ONE worker: background jobs and the model cache live in-process.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from common import config
from data_pipeline.batch_loader import list_batches
from inference import jobs
from inference.health import get_health_status
from inference.model_store import ModelNotAvailable, store
from inference.predict import predict_one
from inference.prediction_logger import log_prediction, prediction_count, recent_predictions
from inference.schemas import (
    INPUT_SCHEMAS, BatchRequest, BreastCancerInput, DiabetesInput, HeartDiseaseInput,
    BatchUpload, PredictionResponse, RollbackResponse,
)
from monitoring.alerting import recent_alerts
from monitoring.monitor import monitoring_history, unchecked_batches
from training import model_registry

log = logging.getLogger("inference")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for disease in config.diseases():  # warm the cache; a missing model must not stop the API
        try:
            store.get(disease)
        except ModelNotAvailable as exc:
            log.warning("%s", exc)
    yield


app = FastAPI(title="Self-Healing MLOps Platform", version="2.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def valid_disease(disease: str) -> str:
    if disease not in config.diseases():
        raise HTTPException(404, f"Unknown disease '{disease}'. Use one of {config.diseases()}")
    return disease


# ── Predictions ────────────────────────────────────────────────────────

def _predict(disease: str, patient) -> dict:
    inputs = patient.model_dump()
    try:
        result = predict_one(disease, inputs)
    except ModelNotAvailable as exc:
        raise HTTPException(503, str(exc))
    try:
        log_prediction(disease, inputs, result)
    except Exception as exc:  # logging must never fail a prediction
        log.warning("Could not log prediction: %s", exc)
    return result


@app.get("/")
def home():
    return {"message": "Self-Healing MLOps Platform API. Visit /docs.", "diseases": config.diseases()}


@app.get("/api/info")
def info():
    """Identifies this service, so the dashboard can tell it is talking to the right API."""
    return {"platform": "self-healing-mlops", "api_version": app.version, "diseases": config.diseases()}


@app.get("/health")
def health():
    return get_health_status()


@app.post("/predict/diabetes", response_model=PredictionResponse)
def predict_diabetes(patient: DiabetesInput):
    return _predict("diabetes", patient)


@app.post("/predict/heart_disease", response_model=PredictionResponse)
def predict_heart_disease(patient: HeartDiseaseInput):
    return _predict("heart_disease", patient)


@app.post("/predict/breast_cancer", response_model=PredictionResponse)
def predict_breast_cancer(patient: BreastCancerInput):
    return _predict("breast_cancer", patient)


# ── Dashboard / operations API ───────────────────────────────────────

def _champion_summary(disease: str) -> dict | None:
    version = model_registry.get_champion_version(disease)
    if version is None:
        return None
    t = version.tags
    return {"version": str(version.version), "algorithm": t.get("algorithm"),
            "test_auc": _num(t.get("test_auc")), "val_auc": _num(t.get("val_auc")),
            "recent_auc": _num(t.get("recent_auc")), "promoted_at": t.get("promoted_at"),
            "training_mode": t.get("training_mode")}


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _status(champion, latest, active_job) -> str:
    if active_job:
        return "retraining"
    if champion is None:
        return "no_model"
    if latest and latest.get("model_version") == champion["version"]:
        if latest.get("performance_dropped"):
            return "degraded"
        if latest.get("drift_detected"):
            return "drift"
    return "healthy"


@app.get("/api/overview")
def overview():
    jobs_now = jobs.list_jobs(50)
    diseases = []
    for d in config.diseases():
        try:
            champion = _champion_summary(d)
        except Exception as exc:
            raise HTTPException(503, f"Model registry unreachable: {exc}")
        history = monitoring_history(d, limit=30)
        latest = history[-1] if history else None
        active = next((j for j in jobs_now if j["target"] == d and j["status"] in ("queued", "running")), None)
        diseases.append({
            "disease": d,
            "display_name": config.disease_settings(d).get("display_name", d),
            "status": _status(champion, latest, active),
            "champion": champion,
            "latest_check": latest,
            "trend": [{"batch": r["batch_name"], "timestamp": r["timestamp"], "drift_share": r["drift_share"],
                       "batch_auc": r["batch_auc"], "baseline_auc": r["baseline_auc"],
                       "drift_detected": r["drift_detected"], "performance_dropped": r["performance_dropped"]}
                      for r in history if r.get("status") == "checked"],
            "batches": len(list_batches(d)),
            "pending_batches": unchecked_batches(d),
            "predictions": prediction_count(d),
            "active_job": active,
        })
    return {
        "diseases": diseases,
        "setup_required": all(x["champion"] is None for x in diseases),
        "busy": jobs.busy(),
        "thresholds": config.settings()["monitoring"] | config.settings()["retraining"],
    }


@app.get("/api/diseases/{disease}/schema")
def disease_schema(disease: str = Depends(valid_disease)):
    schema = INPUT_SCHEMAS[disease].model_json_schema()
    return {"disease": disease, "fields": schema["properties"],
            "required": schema.get("required", []), "example": schema.get("example", {})}


@app.get("/api/models/{disease}/versions")
def model_versions(disease: str = Depends(valid_disease)):
    return {"disease": disease, "versions": model_registry.list_versions(disease),
            "champion_history": model_registry.champion_history(disease)}


@app.post("/api/models/{disease}/rollback", response_model=RollbackResponse)
def rollback(disease: str = Depends(valid_disease)):
    from retraining.rollback import rollback_to_previous

    if jobs.busy():
        raise HTTPException(409, "A background job is running. Try again when it finishes.")
    restored = rollback_to_previous(disease)
    store.invalidate(disease)
    if restored is None:
        return {"disease": disease, "restored_version": None,
                "message": "There is no earlier champion to roll back to."}
    return {"disease": disease, "restored_version": restored,
            "message": f"Version {restored} is live again."}


@app.get("/api/monitoring/history")
def get_monitoring_history(disease: str | None = None, limit: int = Query(100, le=1000)):
    if disease:
        valid_disease(disease)
    return list(reversed(monitoring_history(disease, limit)))


@app.post("/api/monitoring/{disease}/run")
def run_monitoring_check(disease: str = Depends(valid_disease), force: bool = False, heal: bool = True):
    """Checks the newest unchecked batch. With heal=true a retraining job starts when needed."""
    from monitoring.monitor import run_monitoring

    try:
        result = run_monitoring(disease, force=force)
    except LookupError as exc:
        raise HTTPException(409, str(exc))
    job = None
    if heal and result.get("retraining_needed"):
        job = _submit_retraining(disease, result["reasons"])
    return {"monitoring": result, "retraining_job": job}


def _submit_retraining(disease: str, reasons: list[str]) -> dict:
    from retraining.retrain_pipeline import run_retraining

    try:
        return jobs.submit("retraining", disease, run_retraining, disease, reasons,
                           on_done=lambda: store.invalidate(disease))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/retraining/{disease}/run", status_code=202)
def start_retraining(disease: str = Depends(valid_disease), reason: str = "MANUAL"):
    """reason: comma-separated, e.g. DATA_DRIFT,PERFORMANCE_DROP"""
    if model_registry.get_champion_version(disease) is None:
        raise HTTPException(409, f"No champion for {disease} yet. Run setup first.")
    reasons = [r.strip().upper() for r in reason.split(",") if r.strip()] or ["MANUAL"]
    return _submit_retraining(disease, reasons)


@app.get("/api/retraining/history")
def get_retraining_history(disease: str | None = None, limit: int = Query(50, le=500)):
    from retraining.retrain_pipeline import retraining_history

    if disease:
        valid_disease(disease)
    return retraining_history(disease, limit)


@app.get("/api/batches/{disease}")
def get_batches(disease: str = Depends(valid_disease)):
    from common.storage import read_json

    pending = set(unchecked_batches(disease))
    rows = []
    for name in reversed(list_batches(disease)):
        meta = read_json(config.batches_dir(disease) / name / "metadata.json", {}) or {}
        rows.append({"batch_name": name, "pending": name in pending, **meta})
    return rows


@app.post("/api/batches/{disease}", status_code=201)
def create_batch(body: BatchRequest, disease: str = Depends(valid_disease)):
    """Simulates a new production batch (optionally drifted) for demos and testing."""
    from data_pipeline.create_batches import create_batch as make_batch

    if not config.preprocessor_path(disease).exists():
        raise HTTPException(409, "Run setup first: there is no fitted preprocessor yet.")
    return make_batch(disease, body.rows, body.drift, body.label_noise, None, body.concept_shift)


@app.post("/api/batches/{disease}/upload", status_code=201)
def upload_batch(body: BatchUpload, disease: str = Depends(valid_disease)):
    """Registers a CSV file (sent as text by the dashboard) as the next batch."""
    import io

    import pandas as pd

    from data_pipeline.create_batches import BatchFileError, import_batch

    if not config.preprocessor_path(disease).exists():
        raise HTTPException(409, "Run setup first: there is no fitted preprocessor yet.")
    text = body.csv.lstrip("\ufeff")          # Excel adds a BOM
    if not text.strip():
        raise HTTPException(422, "The file is empty.")
    try:  # sep=None detects "," or ";" (Excel in many European locales uses ";")
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        raise HTTPException(422, "The file could not be read as CSV. Save it from Excel as 'CSV (comma delimited)'.")
    try:
        return import_batch(disease, df, body.filename)
    except BatchFileError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/samples")
def list_samples():
    folder = config.path("samples")
    return sorted(p.name for p in folder.glob("*.csv")) if folder.exists() else []


@app.get("/api/samples/{name}")
def get_sample(name: str):
    from fastapi.responses import FileResponse

    file = config.path("samples", name)
    if "/" in name or "\\" in name or not name.endswith(".csv") or not file.exists():
        raise HTTPException(404, "Sample not found")
    return FileResponse(file, media_type="text/csv", filename=name)


@app.get("/api/diseases/{disease}/template")
def csv_template(disease: str = Depends(valid_disease)):
    """Header row plus one example row, for building an upload file."""
    from fastapi.responses import PlainTextResponse

    features = config.features(disease)
    example = INPUT_SCHEMAS[disease].model_json_schema().get("example", {})
    header = ",".join(features + ["Outcome"])
    # Outcome: 1 = disease present, 0 = not present; leave the column out for a drift-only check
    row = ",".join(str(example.get(f, "")) for f in features) + ",1"
    return PlainTextResponse(f"{header}\n{row}\n", media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{disease}_template.csv"'})


@app.get("/api/alerts")
def alerts(disease: str | None = None, limit: int = Query(50, le=500)):
    if disease:
        valid_disease(disease)
    return recent_alerts(limit, disease)


@app.get("/api/predictions/{disease}")
def predictions(disease: str = Depends(valid_disease), limit: int = Query(50, le=500)):
    return recent_predictions(disease, limit)


@app.post("/api/setup", status_code=202)
def setup():
    """Validate data, preprocess, train all diseases, create one clean batch each."""
    from pipeline import setup as run_setup

    try:
        return jobs.submit("setup", None, run_setup, on_done=store.invalidate)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job
