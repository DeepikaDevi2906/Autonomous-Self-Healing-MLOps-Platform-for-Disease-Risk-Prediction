from common import config
from data_pipeline.create_batches import create_batch
from monitoring.drift_detector import detect_drift
from monitoring.monitor import run_monitoring, unchecked_batches


def test_clean_batch_shows_no_drift(trained):
    result = detect_drift("breast_cancer", "batch_001")
    assert result["drift_detected"] is False
    assert (config.batches_dir("breast_cancer") / "batch_001" / "drift_report.html").exists()


def test_strong_drift_is_detected(trained):
    meta = create_batch("heart_disease", drift=2.5, seed=3)
    result = detect_drift("heart_disease", meta["batch_name"])
    assert result["drift_detected"] is True
    assert result["drift_share"] > config.settings()["monitoring"]["drift_share_threshold"]


def test_each_batch_is_checked_only_once(trained):
    create_batch("diabetes", seed=4)
    assert unchecked_batches("diabetes")
    first = run_monitoring("diabetes")
    assert first["status"] == "checked"
    assert unchecked_batches("diabetes") == []
    second = run_monitoring("diabetes")
    assert second["status"] == "no_new_batch" and second["retraining_needed"] is False


def test_monitoring_record_is_json_friendly(trained):
    import json

    create_batch("diabetes", seed=5)
    record = run_monitoring("diabetes")
    json.dumps(record)
    assert isinstance(record["retraining_needed"], bool)
