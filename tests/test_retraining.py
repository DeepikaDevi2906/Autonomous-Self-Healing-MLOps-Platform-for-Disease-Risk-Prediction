from common import config
from data_pipeline.create_batches import create_batch
from monitoring.monitor import run_monitoring
from retraining.retrain_pipeline import run_retraining
from retraining.rollback import rollback_to_previous
from training import model_registry


def test_retraining_registers_challenger_and_follows_rules(trained):
    create_batch("heart_disease", seed=21)
    before = str(model_registry.get_champion_version("heart_disease").version)
    result = run_retraining("heart_disease", ["MANUAL"])
    after = str(model_registry.get_champion_version("heart_disease").version)

    assert result["challenger_version"] != before
    if result["should_promote"]:
        assert after == result["challenger_version"]
    else:
        assert after == before
    tags = {v["version"]: v["tags"] for v in model_registry.list_versions("heart_disease")}
    assert tags[result["challenger_version"]]["status"] in ("promoted", "rejected")


def test_rollback_walks_back_instead_of_ping_pong(trained):
    disease = "diabetes"
    first = str(model_registry.get_champion_version(disease).version)
    # register two more versions and promote each, so history is [first, a, b]
    import mlflow
    from training.train import load_split
    from training.models import TRAINERS

    X, y = load_split(disease, "train")
    config.set_experiment(f"{disease}_retraining")
    promoted = []
    for _ in range(2):
        with mlflow.start_run():
            model = TRAINERS["RandomForest"](X, y)
            v = model_registry.register_model(disease, model, "RandomForest", X.head(3), {"test_auc": 0.8})
        model_registry.promote(disease, v, "test")
        promoted.append(v)

    assert rollback_to_previous(disease) == promoted[0]
    assert rollback_to_previous(disease) == first      # further back, not back to promoted[1]
    assert str(model_registry.get_champion_version(disease).version) == first


def test_cooldown_blocks_immediate_second_retrain(trained):
    disease = "breast_cancer"
    create_batch(disease, seed=31)
    run_monitoring(disease)
    run_retraining(disease, ["MANUAL"])
    create_batch(disease, drift=3.0, seed=32)
    record = run_monitoring(disease)
    assert "DATA_DRIFT" in record["reasons"]
    assert record["cooldown_remaining"] >= 1
    assert record["retraining_needed"] is False
