"""
Simulates a new batch of production data for one disease.

Rows are drawn from the disease's future_pool (rows never used for
training/validation/testing) - without replacement unless more rows are asked
for than the pool holds - lightly jittered, and optionally shifted to
simulate data drift (--drift), given a learnable change in the feature/label
relationship (--concept-shift), or random label noise (--label-noise, which no
model can learn - the platform should refuse to promote on it). The raw batch is then run through the SAME fitted
preprocessor used for training.

    python src/data_pipeline/create_batches.py --disease diabetes
    python src/data_pipeline/create_batches.py --disease diabetes --drift 1.5
    python src/data_pipeline/create_batches.py --disease diabetes --concept-shift 0.8
    python src/data_pipeline/create_batches.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from common import config
from common.storage import utc_now, write_json
from data_pipeline.batch_loader import list_batches, batch_number

JITTER = 0.03  # 3% of a feature's std, so bootstrapped rows are not exact copies


def next_batch_name(disease: str) -> str:
    existing = list_batches(disease, processed_only=False)
    n = batch_number(existing[-1]) + 1 if existing else 1
    return f"batch_{n:03d}"


def simulate_raw_batch(
    disease: str, n_rows: int | None = None, drift: float = 0.0, label_noise: float = 0.0,
    seed: int | None = None, concept_shift: float = 0.0,
) -> pd.DataFrame:
    pool_file = config.processed_dir(disease) / "future_pool_raw.csv"
    if not pool_file.exists():
        raise FileNotFoundError(f"{pool_file} not found. Run preprocessing first.")
    pool = pd.read_csv(pool_file)
    rng = np.random.default_rng(seed)

    pool["source_id"] = pool.index
    n_rows = n_rows or len(pool)
    batch = pool.sample(n=n_rows, replace=n_rows > len(pool),
                        random_state=int(rng.integers(1e9))).reset_index(drop=True)
    feats = config.features(disease)
    drift_feats = set(config.disease_settings(disease).get("drift_features", []))

    for col in feats:
        values = batch[col].astype(float)
        is_integer = np.allclose(pool[col].dropna() % 1, 0)
        std = float(pool[col].std() or 0.0)
        if not is_integer and std > 0:
            values = values + rng.normal(0, JITTER * std, len(values))
        if drift and col in drift_feats and std > 0:
            values = values + drift * std
        if pool[col].min() >= 0:
            values = values.clip(lower=0)
        batch[col] = values.round(0) if is_integer else values.round(4)

    if concept_shift > 0 and drift_feats:
        # patients above the median of the first drift feature now have the
        # opposite outcome with probability `concept_shift` - a learnable pattern
        key = config.disease_settings(disease)["drift_features"][0]
        affected = (batch[key] > pool[key].median()) & (rng.random(len(batch)) < concept_shift)
        batch.loc[affected, "Outcome"] = 1 - batch.loc[affected, "Outcome"]

    if label_noise > 0:
        flip = rng.random(len(batch)) < label_noise
        batch.loc[flip, "Outcome"] = 1 - batch.loc[flip, "Outcome"]

    batch["Outcome"] = batch["Outcome"].astype(int)
    # source_id lets retraining keep a batch's evaluation rows out of training
    return batch[["source_id"] + feats + ["Outcome"]]


def process_batch(disease: str, batch_name: str) -> Path:
    """raw_data.csv -> processed_data.csv using the training preprocessor."""
    folder = config.batches_dir(disease) / batch_name
    raw = pd.read_csv(folder / "raw_data.csv")
    preprocessor = joblib.load(config.preprocessor_path(disease))
    processed = preprocessor.transform(raw)
    if "Outcome" in raw.columns:
        processed["Outcome"] = raw["Outcome"].astype(int).values
    if "source_id" in raw.columns:
        processed.insert(0, "source_id", raw["source_id"].values)
    out = folder / "processed_data.csv"
    processed.to_csv(out, index=False)
    return out


def create_batch(
    disease: str, n_rows: int | None = None, drift: float = 0.0, label_noise: float = 0.0,
    seed: int | None = None, concept_shift: float = 0.0,
) -> dict:
    name = next_batch_name(disease)
    folder = config.batches_dir(disease) / name
    folder.mkdir(parents=True, exist_ok=True)

    raw = simulate_raw_batch(disease, n_rows, drift, label_noise, seed, concept_shift)
    raw.to_csv(folder / "raw_data.csv", index=False)
    process_batch(disease, name)

    meta = {
        "disease": disease, "batch_name": name, "rows": len(raw), "source": "simulated", "labelled": True,
        "simulated_drift": drift, "simulated_label_noise": label_noise,
        "simulated_concept_shift": concept_shift,
        "created_at": utc_now(),
    }
    write_json(folder / "metadata.json", meta)
    print(f"Created {name} for {disease}: {len(raw)} rows "
          f"(drift={drift}, concept_shift={concept_shift}, label_noise={label_noise})")
    return meta


class BatchFileError(ValueError):
    """A problem with an uploaded batch file, phrased for the person who uploaded it."""


MAX_UPLOAD_ROWS = 20_000


def _rows(mask: pd.Series, limit: int = 5) -> str:
    """Spreadsheet row numbers (header = row 1) of the first failing rows."""
    rows = [str(i + 2) for i in mask[mask].index[:limit]]
    more = int(mask.sum()) - len(rows)
    return ", ".join(rows) + (f" and {more} more" if more > 0 else "")


def _value_limits(disease: str) -> dict:
    """Same limits as the prediction form (inference/schemas.py)."""
    from inference.schemas import INPUT_SCHEMAS

    props = INPUT_SCHEMAS[disease].model_json_schema()["properties"]
    return {k: (v.get("minimum"), v.get("maximum")) for k, v in props.items()}


def import_batch(disease: str, df: pd.DataFrame, filename: str = "upload.csv") -> dict:
    """
    Registers an uploaded CSV as the next batch for a disease.

    * Every feature column is required; names may differ in case or spacing.
      Extra columns are ignored.
    * Outcome is optional (1 = disease present, 0 = not present). Without it the
      batch gets a drift check only and is never used for retraining.
    * Empty cells are allowed and are imputed exactly as in training.
    Raises BatchFileError with a message that says exactly what to fix.
    """
    feats = config.features(disease)
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    wanted = {f.lower(): f for f in feats + ["Outcome"]}
    df = df.rename(columns={orig: wanted[key] for key, orig in lookup.items() if key in wanted})

    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise BatchFileError(f"Missing column(s): {', '.join(missing)}. A {disease} file needs: "
                             f"{', '.join(feats)} (and optionally Outcome). Download the template to get them right.")

    df = df.dropna(how="all").reset_index(drop=True)
    min_rows = config.settings()["monitoring"]["min_batch_rows"]
    if len(df) < min_rows:
        raise BatchFileError(f"The file has {len(df)} patient rows; at least {min_rows} are needed for reliable statistics.")
    if len(df) > MAX_UPLOAD_ROWS:
        raise BatchFileError(f"The file has {len(df)} rows; the maximum is {MAX_UPLOAD_ROWS:,}.")

    out = pd.DataFrame(index=df.index)
    problems = []
    limits = _value_limits(disease)
    for col in feats:
        values = pd.to_numeric(df[col], errors="coerce")
        not_numeric = values.isna() & df[col].notna() & (df[col].astype(str).str.strip() != "")
        if not_numeric.any():
            problems.append(f"'{col}' has text that is not a number in row(s) {_rows(not_numeric)}")
            continue
        if values.isna().all():
            problems.append(f"'{col}' is empty in every row")
            continue
        lo, hi = limits.get(col, (None, None))
        out_of_range = (values < lo if lo is not None else False) | (values > hi if hi is not None else False)
        if isinstance(out_of_range, pd.Series) and out_of_range.any():
            problems.append(f"'{col}' must be between {lo} and {hi}; see row(s) {_rows(out_of_range)}")
        out[col] = values

    labelled = "Outcome" in df.columns and df["Outcome"].notna().any()
    warnings = []
    if labelled:
        outcome = pd.to_numeric(df["Outcome"], errors="coerce")
        bad = outcome.isna() | ~outcome.isin([0, 1])
        if bad.any():
            problems.append(f"'Outcome' must be 0 or 1 in every row (1 = disease present); see row(s) {_rows(bad)}. "
                            "Or remove the column for a drift-only check")
        else:
            out["Outcome"] = outcome.astype(int)
            if outcome.nunique() < 2:
                warnings.append("Outcome has only one value, so accuracy (AUC) cannot be measured for this batch")
    else:
        warnings.append("No Outcome column: this batch gets a drift check only and is not used for retraining")

    if problems:
        raise BatchFileError("Please fix the file: " + "; ".join(problems) + ".")

    empty_cells = int(out[feats].isna().sum().sum())
    if empty_cells:
        warnings.append(f"{empty_cells} empty cell(s) will be filled with training medians")

    name = next_batch_name(disease)
    folder = config.batches_dir(disease) / name
    folder.mkdir(parents=True, exist_ok=True)
    # unique negative ids keep uploaded rows apart from simulated (pool) rows
    out.insert(0, "source_id", [-(batch_number(name) * 100_000 + i + 1) for i in range(len(out))])
    out.to_csv(folder / "raw_data.csv", index=False)
    process_batch(disease, name)
    meta = {
        "disease": disease, "batch_name": name, "rows": int(len(out)), "source": "upload",
        "filename": str(filename)[:120], "labelled": bool(labelled), "warnings": warnings,
        "created_at": utc_now(),
    }
    write_json(folder / "metadata.json", meta)
    print(f"Imported {filename} as {name} for {disease}: {len(out)} rows, labelled={labelled}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--disease", choices=config.diseases())
    parser.add_argument("--all", action="store_true", help="create one batch for every disease")
    parser.add_argument("--rows", type=int, default=None, help="default: size of the future pool")
    parser.add_argument("--drift", type=float, default=0.0, help="shift in standard deviations")
    parser.add_argument("--label-noise", type=float, default=0.0, help="fraction of labels to flip")
    parser.add_argument("--concept-shift", type=float, default=0.0, help="learnable label change, 0-1")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.all and not args.disease:
        parser.error("pass --disease <name> or --all")
    for d in (config.diseases() if args.all else [args.disease]):
        create_batch(d, args.rows, args.drift, args.label_noise, args.seed, args.concept_shift)


if __name__ == "__main__":
    main()
