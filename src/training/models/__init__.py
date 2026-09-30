"""Registry of candidate algorithms. Add a new one here and it is trained everywhere."""

from training.models import neural_network, random_forest, xgboost_model

TRAINERS = {
    "RandomForest": random_forest.train,
    "XGBoost": xgboost_model.train,
    "NeuralNet": neural_network.train,
}
