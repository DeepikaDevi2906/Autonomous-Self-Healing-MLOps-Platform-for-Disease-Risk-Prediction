"""Scores one patient with the cached champion for a disease."""

from __future__ import annotations

import pandas as pd

from inference.model_store import store


def predict_one(disease: str, input_data: dict) -> dict:
    loaded = store.get(disease)
    raw = pd.DataFrame([input_data])
    X = loaded.preprocessor.transform(raw)

    probability = float(loaded.model.predict_proba(X)[0][1])
    prediction = int(probability >= 0.5)
    return {
        "disease": disease,
        "prediction": prediction,
        "prediction_label": "Disease detected" if prediction else "No disease detected",
        "probability": round(probability, 4),
        "model_version": loaded.version,
        "model_algorithm": loaded.algorithm,
    }
