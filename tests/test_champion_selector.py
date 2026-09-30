import pytest

from training.champion_selector import select_champion


def test_picks_highest_score():
    assert select_champion({"RandomForest": 0.84, "XGBoost": 0.82, "NeuralNet": 0.86}) == "NeuralNet"


def test_skips_models_without_a_score():
    assert select_champion({"RandomForest": None, "XGBoost": 0.7}) == "XGBoost"


def test_tie_goes_to_first_listed():
    assert select_champion({"RandomForest": 0.9, "XGBoost": 0.9}) == "RandomForest"


def test_no_valid_scores_raises():
    with pytest.raises(ValueError):
        select_champion({"RandomForest": None})
