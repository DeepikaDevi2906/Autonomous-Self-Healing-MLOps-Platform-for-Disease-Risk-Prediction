"""Loads the combined raw CSV and checks it has the columns the platform needs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import config

REQUIRED_COLUMNS = ["disease_type", "Outcome"]


def load_data(file_path: str | Path | None = None) -> pd.DataFrame:
    file_path = Path(file_path) if file_path else config.path(config.settings()["data"]["raw_path"])
    if not file_path.is_absolute():
        file_path = config.path(str(file_path))

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"{file_path} was read but has no rows.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    print(f"Loaded {len(df)} rows from {file_path}")
    print(f"Diseases in this file: {sorted(df['disease_type'].dropna().unique().tolist())}")
    return df


if __name__ == "__main__":
    print(load_data().head())
