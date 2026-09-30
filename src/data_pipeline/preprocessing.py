"""
Prepares ONE disease for training.

    1. Select that disease's rows and configured feature columns
    2. Make Outcome consistent (1 = disease present) where the source data is inverted
    3. Stratified split: train / val / test / future_pool
    4. Fit DiseasePreprocessor (missing-code handling, median imputation, scaling)
       on TRAIN ONLY, then apply it to val and test
    5. Save processed splits, the fitted preprocessor, reference data for drift
       detection, and the raw future_pool used to simulate production batches
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from common import config
from data_pipeline.transformers import DiseasePreprocessor


def filter_by_disease(df: pd.DataFrame, disease: str) -> pd.DataFrame:
    cols = config.features(disease)
    subset = df.loc[df["disease_type"] == disease, cols + ["Outcome"]].copy()
    subset["Outcome"] = subset["Outcome"].astype(int)
    if config.disease_settings(disease).get("invert_outcome"):
        subset["Outcome"] = 1 - subset["Outcome"]
        print("  Inverted Outcome so that 1 = disease present")
    print(f"Filtered to {disease}: {subset.shape[0]} rows, {len(cols)} features")
    return subset.reset_index(drop=True)


def split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    split = config.settings()["data"]["split"]
    seed = config.settings()["data"]["random_state"]
    total = sum(split.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"data.split fractions must add up to 1.0 (got {total})")

    train, rest = train_test_split(
        df, train_size=split["train"], random_state=seed, stratify=df["Outcome"]
    )
    remaining = 1.0 - split["train"]
    val, rest = train_test_split(
        rest, train_size=split["val"] / remaining, random_state=seed, stratify=rest["Outcome"]
    )
    remaining -= split["val"]
    test, pool = train_test_split(
        rest, train_size=split["test"] / remaining, random_state=seed, stratify=rest["Outcome"]
    )
    parts = {"train": train, "val": val, "test": test, "future_pool": pool}
    print("Split sizes -> " + ", ".join(f"{k}: {len(v)}" for k, v in parts.items()))
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def to_processed(preprocessor: DiseasePreprocessor, raw: pd.DataFrame) -> pd.DataFrame:
    processed = preprocessor.transform(raw)
    processed["Outcome"] = raw["Outcome"].astype(int).values
    return processed


def run_preprocessing(df: pd.DataFrame, disease: str) -> dict:
    print(f"\n--- Preprocessing {disease} ---")
    disease_df = filter_by_disease(df, disease)
    parts = split_data(disease_df)

    preprocessor = DiseasePreprocessor(
        disease,
        config.features(disease),
        config.disease_settings(disease).get("missing_codes", {}),
    ).fit(parts["train"])

    out = config.processed_dir(disease)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        to_processed(preprocessor, parts[name]).to_csv(out / f"{name}.csv", index=False)
    parts["future_pool"].to_csv(out / "future_pool_raw.csv", index=False)
    joblib.dump(preprocessor, config.preprocessor_path(disease))

    ref = config.reference_path(disease)
    ref.parent.mkdir(parents=True, exist_ok=True)
    to_processed(preprocessor, parts["train"]).drop(columns=["Outcome"]).to_csv(ref, index=False)

    print(f"Saved processed data and preprocessor to {out}")
    print(f"--- Done with {disease} ---")
    return {k: len(v) for k, v in parts.items()}


def run_all(df: pd.DataFrame | None = None) -> dict:
    from data_pipeline.ingestion import load_data
    from data_pipeline.validation import validate

    df = df if df is not None else load_data()
    validate(df)
    return {d: run_preprocessing(df, d) for d in config.diseases()}


if __name__ == "__main__":
    run_all()
