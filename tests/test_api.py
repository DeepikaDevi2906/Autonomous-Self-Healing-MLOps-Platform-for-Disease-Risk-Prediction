import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained):
    from inference.main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_every_disease(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert set(body["models"]) == {"diabetes", "heart_disease", "breast_cancer"}


@pytest.mark.parametrize("disease", ["diabetes", "heart_disease", "breast_cancer"])
def test_predict_with_schema_example(client, disease):
    example = client.get(f"/api/diseases/{disease}/schema").json()["example"]
    response = client.post(f"/predict/{disease}", json=example)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction"] in (0, 1) and 0 <= body["probability"] <= 1
    assert body["model_version"]
    logged = client.get(f"/api/predictions/{disease}").json()
    assert logged and logged[0]["prediction"] == str(body["prediction"])


def test_invalid_input_is_rejected(client):
    example = client.get("/api/diseases/diabetes/schema").json()["example"]
    assert client.post("/predict/diabetes", json={**example, "Glucose": -5}).status_code == 422
    assert client.post("/predict/diabetes", json={**example, "extra": 1}).status_code == 422


def test_unknown_disease_is_404(client):
    assert client.get("/api/models/flu/versions").status_code == 404


def test_overview_and_batches(client):
    body = client.get("/api/overview").json()
    assert body["setup_required"] is False
    assert {d["disease"] for d in body["diseases"]} == {"diabetes", "heart_disease", "breast_cancer"}

    created = client.post("/api/batches/breast_cancer", json={"drift": 0}).json()
    assert created["batch_name"].startswith("batch_")
    run = client.post("/api/monitoring/breast_cancer/run?heal=false").json()
    assert run["monitoring"]["batch_name"] == created["batch_name"]


def test_rollback_without_history_is_explained(client):
    body = client.post("/api/models/heart_disease/rollback").json()
    assert "message" in body
