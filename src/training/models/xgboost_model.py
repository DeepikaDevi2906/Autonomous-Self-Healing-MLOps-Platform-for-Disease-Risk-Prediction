"""XGBoost: trees built one after another, each fixing the previous ones' mistakes."""

from xgboost import XGBClassifier

from common import config


def train(X_train, y_train):
    params = config.settings()["models"]["xgboost"]
    model = XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss", n_jobs=1, **params)
    return model.fit(X_train, y_train)
