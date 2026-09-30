"""A small multi-layer perceptron."""

from sklearn.neural_network import MLPClassifier

from common import config


def train(X_train, y_train):
    params = dict(config.settings()["models"]["neural_network"])
    params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    model = MLPClassifier(random_state=42, **params)
    return model.fit(X_train, y_train)
