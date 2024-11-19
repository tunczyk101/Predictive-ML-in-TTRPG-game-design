from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from training.constants import RANDOM_STATE
from training.models.Ordinal_Classifier.Ordinal_Classifier import OrdinalClassifier


class SimpleOrdinalClassification(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.default_params = {
            "max_features": 0.3,
            "n_estimators": 100,
            "criterion": "gini",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        self.default_params.update(kwargs)

        for param, value in self.default_params.items():
            setattr(self, param, value)

        self.base_model_ = RandomForestClassifier(**self.default_params)
        self.model_ = OrdinalClassifier(self.base_model_)

    def fit(self, X, y):
        self.model_.fit(X, y)

    def predict(self, X):
        return self.model_.predict(X)

    def get_params(self, deep=True):
        return self.base_model_.get_params(deep)

    def set_params(self, **params):
        self.base_model_.set_params(**params)

        self.model_ = OrdinalClassifier(self.base_model_)

        return self
