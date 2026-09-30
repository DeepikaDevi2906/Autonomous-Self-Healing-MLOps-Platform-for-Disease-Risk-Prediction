"""
Data validation, run before preprocessing (the old version was never called
and imported a module, ingestion_v2, that did not exist).

Hard failures raise ValueError; soft problems are returned as warnings.
"""

from __future__ import annotations

import pandas as pd

from common import config

MISSING_WARNING_PERCENT = 20


def check_outcome_column(df: pd.DataFrame) -> None:
    if df["Outcome"].isna().any():
        raise ValueError(f"Outcome column has {int(df['Outcome'].isna().sum())} missing value(s)")
    invalid = sorted(set(df["Outcome"].unique()) - {0, 1})
    if invalid:
        raise ValueError(f"Outcome column has invalid value(s): {invalid}")


def check_disease(df: pd.DataFrame, disease: str) -> dict:
    subset = df[df["disease_type"] == disease]
    if subset.empty:
        raise ValueError(f"No rows found for disease '{disease}'")

    feature_cols = config.features(disease)
    missing_cols = [c for c in feature_cols if c not in subset.columns]
    if missing_cols:
        raise ValueError(f"{disease}: missing feature column(s) {missing_cols}")

    warnings = []
    missing_counts = subset[feature_cols].isna().sum()
    for col, count in missing_counts.items():
        percent = count / len(subset) * 100
        if percent == 100:
            raise ValueError(f"{disease}: column '{col}' is completely empty")
        if percent > MISSING_WARNING_PERCENT:
            warnings.append(f"'{col}' is {percent:.1f}% missing")

    for col, codes in config.disease_settings(disease).get("missing_codes", {}).items():
        n = int(subset[col].isin(codes).sum())
        if n:
            warnings.append(f"'{col}' has {n} placeholder value(s) {codes}; they will be imputed")

    classes = subset["Outcome"].value_counts().to_dict()
    if len(classes) < 2:
        raise ValueError(f"{disease}: Outcome has only one class {classes}")

    return {
        "disease": disease,
        "rows": int(len(subset)),
        "missing_values": int(missing_counts.sum()),
        "class_counts": {int(k): int(v) for k, v in classes.items()},
        "warnings": warnings,
    }


def validate(df: pd.DataFrame) -> dict:
    print("Running validation checks...")
    check_outcome_column(df)

    unknown = sorted(set(df["disease_type"].dropna()) - set(config.diseases()))
    report = {"unknown_diseases": unknown, "diseases": {}}
    if unknown:
        print(f"  Warning: rows for unconfigured disease(s) will be ignored: {unknown}")

    for disease in config.diseases():
        result = check_disease(df, disease)
        report["diseases"][disease] = result
        print(f"  {disease}: {result['rows']} rows, {result['missing_values']} missing values")
        for w in result["warnings"]:
            print(f"    Warning: {w}")

    print("Validation complete.")
    return report


if __name__ == "__main__":
    from data_pipeline.ingestion import load_data

    validate(load_data())
