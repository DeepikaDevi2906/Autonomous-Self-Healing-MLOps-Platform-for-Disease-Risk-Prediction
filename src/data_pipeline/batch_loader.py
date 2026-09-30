"""
Finds and loads production batches:

    data/batches/<disease>/batch_001/raw_data.csv        (raw patient values)
    data/batches/<disease>/batch_001/processed_data.csv  (after DiseasePreprocessor)
    data/batches/<disease>/batch_001/metadata.json

Every batch is split deterministically into a training part and an evaluation
part. Retraining only learns from the training parts; the evaluation parts of the
newest batches are used to compare champion and challenger on RECENT data.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.model_selection import train_test_split

from common import config

BATCH_PATTERN = re.compile(r"^batch_(\d+)$")


def batch_number(name: str) -> int:
    match = BATCH_PATTERN.match(name)
    if not match:
        raise ValueError(f"Not a batch folder name: {name}")
    return int(match.group(1))


def list_batches(disease: str, processed_only: bool = True) -> list[str]:
    """Batch names sorted numerically (batch_2 before batch_10)."""
    root = config.batches_dir(disease)
    if not root.exists():
        return []
    names = [
        p.name for p in root.iterdir()
        if p.is_dir() and BATCH_PATTERN.match(p.name)
        and (not processed_only or (p / "processed_data.csv").exists())
    ]
    return sorted(names, key=batch_number)


def latest_batch(disease: str) -> str | None:
    batches = list_batches(disease)
    return batches[-1] if batches else None


def load_batch(disease: str, batch_name: str) -> pd.DataFrame:
    file = config.batches_dir(disease) / batch_name / "processed_data.csv"
    if not file.exists():
        raise FileNotFoundError(f"No processed batch at {file}")
    return pd.read_csv(file)


def split_batch(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fraction = config.settings()["retraining"]["batch_eval_fraction"]
    seed = config.settings()["data"]["random_state"]
    stratify = batch_df["Outcome"] if batch_df["Outcome"].value_counts().min() >= 2 else None
    train_part, eval_part = train_test_split(
        batch_df, test_size=fraction, random_state=seed, stratify=stratify
    )
    return train_part.reset_index(drop=True), eval_part.reset_index(drop=True)


def load_for_retraining(disease: str) -> dict:
    """
    Returns:
        train          historical train + training part of every batch
        recent_eval    held-out parts of the newest batches (None if no batches)
        batches_used   list of batch names
    """
    historical_path = config.processed_dir(disease) / "train.csv"
    if not historical_path.exists():
        raise FileNotFoundError(
            f"No historical training data at {historical_path}. Run preprocessing first."
        )
    historical = pd.read_csv(historical_path)
    print(f"Loaded historical data: {len(historical)} rows")

    # Only labelled batches can be learned from (uploads may have no Outcome)
    batches = [b for b in list_batches(disease) if "Outcome" in load_batch(disease, b).columns]
    parts = [split_batch(load_batch(disease, name)) for name in batches]

    # Recent data = held-out parts of the newest N batches. Simulated batches can
    # re-use rows of the same source pool, so duplicates keep their newest copy.
    n_recent = config.settings()["retraining"].get("recent_eval_batches", 1)
    recent_parts = [eval_part for _, eval_part in parts[-n_recent:]]
    recent_eval = pd.concat(recent_parts, ignore_index=True) if recent_parts else None
    if recent_eval is not None and "source_id" in recent_eval.columns:
        # uploaded rows have no source_id; only de-duplicate simulated rows
        has_id = recent_eval["source_id"].notna()
        recent_eval = pd.concat([recent_eval[~has_id],
                                 recent_eval[has_id].drop_duplicates("source_id", keep="last")],
                                ignore_index=True)

    # Rows in the recent evaluation set must never be trained on, in ANY batch.
    held_out_ids = set()
    if recent_eval is not None and "source_id" in recent_eval.columns:
        held_out_ids = set(recent_eval["source_id"].dropna())

    frames = [historical]
    for name, (train_part, eval_part) in zip(batches, parts):
        if held_out_ids and "source_id" in train_part.columns:
            train_part = train_part[~train_part["source_id"].isin(held_out_ids)]
        frames.append(train_part.drop(columns=["source_id"], errors="ignore"))
        print(f"Loaded {name}: {len(train_part)} train rows, {len(eval_part)} held-out rows")

    if not batches:
        print(f"No batches found for {disease}. Using historical data only.")
    if recent_eval is not None:
        recent_eval = recent_eval.drop(columns=["source_id"], errors="ignore")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined training rows for {disease}: {len(combined)}")
    return {"train": combined, "recent_eval": recent_eval, "batches_used": batches}


if __name__ == "__main__":
    print(load_for_retraining("diabetes")["train"].head())
