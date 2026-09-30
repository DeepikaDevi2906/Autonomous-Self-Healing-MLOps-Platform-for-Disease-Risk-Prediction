from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture(scope="module")
def client(trained):
    from inference.main import app

    with TestClient(app) as c:
        yield c


def upload(client, disease, name, text=None):
    csv = text if text is not None else (SAMPLES / name).read_text()
    return client.post(f"/api/batches/{disease}/upload", json={"filename": name, "csv": csv})


def test_labelled_upload_is_monitored(client):
    r = upload(client, "diabetes", "diabetes_drifted.csv")
    assert r.status_code == 201, r.text
    meta = r.json()
    assert meta["labelled"] is True and meta["rows"] == 78
    check = client.post("/api/monitoring/diabetes/run?heal=false").json()["monitoring"]
    assert check["batch_name"] == meta["batch_name"]
    assert check["drift_detected"] is True
    assert check["batch_auc"] is not None


def test_unlabelled_upload_gets_drift_check_only(client):
    r = upload(client, "breast_cancer", "breast_cancer_unlabelled.csv")
    assert r.status_code == 201, r.text
    assert r.json()["labelled"] is False
    check = client.post("/api/monitoring/breast_cancer/run?heal=false").json()["monitoring"]
    assert check["labelled"] is False and check["batch_auc"] is None
    assert check["performance_dropped"] is False
    # retraining still works: unlabelled batches are simply not trained on
    from data_pipeline.batch_loader import load_for_retraining
    data = load_for_retraining("breast_cancer")
    assert check["batch_name"] not in data["batches_used"]


def test_semicolon_csv_and_extra_columns_are_accepted(client):
    df = pd.read_csv(SAMPLES / "heart_disease_clean.csv")
    df.insert(0, "patient_name", "x")
    r = upload(client, "heart_disease", "excel.csv", df.to_csv(index=False, sep=";"))
    assert r.status_code == 201, r.text


@pytest.mark.parametrize("change, message", [
    (lambda df: df.drop(columns=["Glucose"]), "Missing column"),
    (lambda df: df.head(10), "at least 30"),
    (lambda df: df.assign(Outcome=2), "must be 0 or 1"),
    (lambda df: df.assign(BMI="high"), "is not a number"),
    (lambda df: df.assign(Glucose=-5), "must be between"),
    (lambda df: df.assign(Age=float("nan")), "is empty in every row"),
])
def test_invalid_files_are_explained(client, change, message):
    df = change(pd.read_csv(SAMPLES / "diabetes_clean.csv"))
    r = upload(client, "diabetes", "bad.csv", df.to_csv(index=False))
    assert r.status_code == 422
    assert message in r.json()["detail"]


def test_samples_are_served(client):
    names = client.get("/api/samples").json()
    assert "diabetes_clean.csv" in names and len(names) == 12
    r = client.get("/api/samples/diabetes_clean.csv")
    assert r.status_code == 200 and r.text.startswith("Pregnancies,")
    assert client.get("/api/samples/..%2Fconfig%2Fsettings.yaml").status_code == 404


def test_headers_are_matched_loosely_and_bom_is_ignored(client):
    df = pd.read_csv(SAMPLES / "diabetes_clean.csv")
    df.columns = [f"  {c.upper()} " for c in df.columns]
    r = upload(client, "diabetes", "messy.csv", "\ufeff" + df.to_csv(index=False))
    assert r.status_code == 201, r.text


def test_unlabelled_drift_alerts_but_does_not_retrain(client):
    df = pd.read_csv(SAMPLES / "heart_disease_drifted.csv").drop(columns=["Outcome"])
    r = upload(client, "heart_disease", "drift_no_outcome.csv", df.to_csv(index=False))
    assert r.status_code == 201 and r.json()["labelled"] is False
    body = client.post("/api/monitoring/heart_disease/run?heal=true").json()
    check = body["monitoring"]
    assert "DATA_DRIFT" in check["reasons"]
    assert check["retraining_needed"] is False and body["retraining_job"] is None
    assert "outcomes" in check["note"]


def test_template_has_the_right_columns(client):
    from common import config

    r = client.get("/api/diseases/breast_cancer/template")
    assert r.status_code == 200
    header = r.text.splitlines()[0].split(",")
    assert header == config.features("breast_cancer") + ["Outcome"]
