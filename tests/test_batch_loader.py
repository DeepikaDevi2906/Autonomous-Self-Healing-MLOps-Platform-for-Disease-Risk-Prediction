import pandas as pd

from common import config
from data_pipeline.batch_loader import list_batches, load_batch, load_for_retraining
from data_pipeline.create_batches import create_batch


def test_batches_sort_numerically(project, trained):
    root = config.batches_dir("heart_disease")
    for name in ("batch_2", "batch_10"):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "processed_data.csv").write_text("x\n1\n")
    try:
        names = list_batches("heart_disease")
        assert names.index("batch_2") < names.index("batch_10")
    finally:
        for name in ("batch_2", "batch_10"):
            (root / name / "processed_data.csv").unlink()
            (root / name).rmdir()


def test_batches_never_contain_test_rows(trained):
    pool = pd.read_csv(config.processed_dir("diabetes") / "future_pool_raw.csv")
    batch = pd.read_csv(config.batches_dir("diabetes") / "batch_001" / "raw_data.csv")
    assert batch["source_id"].max() < len(pool)  # every row comes from the held-out pool


def test_recent_eval_rows_are_never_trained_on(trained):
    create_batch("breast_cancer", seed=11)
    data = load_for_retraining("breast_cancer")
    eval_rows = data["recent_eval"]
    assert eval_rows is not None and "source_id" not in eval_rows.columns

    newest = load_batch("breast_cancer", data["batches_used"][-1])
    feats = config.features("breast_cancer")
    eval_keys = set(map(tuple, eval_rows[feats].round(6).values))
    trained_keys = set(map(tuple, data["train"][feats].round(6).values))
    assert not eval_keys & trained_keys
    assert len(newest) > len(eval_rows)
