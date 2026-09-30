import joblib
import numpy as np
import pandas as pd

from common import config
from data_pipeline.ingestion import load_data
from data_pipeline.preprocessing import filter_by_disease, split_data
from data_pipeline.transformers import DiseasePreprocessor


def test_splits_are_disjoint_and_complete(project):
    df = filter_by_disease(load_data(), "diabetes")
    df["row_id"] = range(len(df))
    parts = split_data(df)
    ids = [set(p["row_id"]) for p in parts.values()]
    assert sum(len(i) for i in ids) == len(df)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert not a & b


def test_heart_outcome_is_inverted_so_1_means_disease(project):
    raw = load_data()
    original = raw.loc[raw.disease_type == "heart_disease", "Outcome"].astype(int).values
    filtered = filter_by_disease(raw, "heart_disease")
    assert (filtered["Outcome"].values == 1 - original).all()
    # patients with disease should have more blocked vessels (ca) on average
    assert filtered.groupby("Outcome")["ca"].mean()[1] > filtered.groupby("Outcome")["ca"].mean()[0]


def test_zero_insulin_is_imputed_at_prediction_time(project):
    train = pd.DataFrame({f: [1.0, 2.0, 3.0, 4.0] for f in config.features("diabetes")})
    train["Insulin"] = [100.0, 200.0, 0.0, 300.0]
    prep = DiseasePreprocessor("diabetes", config.features("diabetes"), {"Insulin": [0]}).fit(train)
    assert prep.medians_["Insulin"] == 200.0  # median of the non-zero values only

    request = train.iloc[[0]].copy()
    request["Insulin"] = 0.0
    median_row = train.iloc[[0]].copy()
    median_row["Insulin"] = 200.0
    assert np.allclose(prep.transform(request).values, prep.transform(median_row).values)


def test_saved_preprocessor_matches_processed_train(trained):
    prep = joblib.load(config.preprocessor_path("diabetes"))
    train = pd.read_csv(config.processed_dir("diabetes") / "train.csv")
    assert list(prep.features) == [c for c in train.columns if c != "Outcome"]
    assert abs(train.drop(columns="Outcome").mean()).max() < 1e-6  # scaled on train
