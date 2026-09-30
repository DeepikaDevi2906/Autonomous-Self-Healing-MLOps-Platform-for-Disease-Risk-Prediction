"""
DiseasePreprocessor - the ONE place where raw patient values become model inputs.

It is fitted on training rows only and then reused, unchanged, for validation,
test, new batches and live API requests. That removes the old train/serve skew
(e.g. Insulin=0 was fixed during training but not at prediction time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DiseasePreprocessor:
    def __init__(self, disease: str, features: list[str], missing_codes: dict | None = None):
        self.disease = disease
        self.features = list(features)
        self.missing_codes = {k: list(v) for k, v in (missing_codes or {}).items()}

    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise ValueError(f"{self.disease}: missing feature column(s) {missing}")
        X = X[self.features].apply(pd.to_numeric, errors="coerce").astype(float)
        for col, codes in self.missing_codes.items():
            if col in X.columns:
                X.loc[X[col].isin(codes), col] = np.nan
        return X

    def fit(self, X: pd.DataFrame) -> "DiseasePreprocessor":
        X = self._clean(X)
        self.medians_ = X.median()
        if self.medians_.isna().any():
            empty = self.medians_[self.medians_.isna()].index.tolist()
            raise ValueError(f"{self.disease}: columns with no usable values: {empty}")
        self.scaler_ = StandardScaler().fit(X.fillna(self.medians_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("DiseasePreprocessor must be fitted before transform()")
        X = self._clean(X).fillna(self.medians_)
        scaled = self.scaler_.transform(X)
        return pd.DataFrame(scaled, columns=self.features, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
