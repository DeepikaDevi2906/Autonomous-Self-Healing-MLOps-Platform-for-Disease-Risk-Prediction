"""Random Forest: many decision trees that vote on the answer."""

from sklearn.ensemble import RandomForestClassifier

from common import config


def train(X_train, y_train):
    params = config.settings()["models"]["random_forest"]
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    return model.fit(X_train, y_train)
