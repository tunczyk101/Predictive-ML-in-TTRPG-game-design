from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from training.constants import RANDOM_STATE
from training.models.Ordinal_Classifier.Ordinal_Classifier import OrdinalClassifier


class SimpleOrdinalClassification(BaseEstimator, ClassifierMixin):
    def __init__(self, max_features=0.3, n_estimators=100, criterion="gini"):
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.model_ = OrdinalClassifier(
            RandomForestClassifier(
                max_features=self.max_features,
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        )

    def fit(self, X, y):
        self.model_.fit(X, y)

    def predict(self, X):
        return self.model_.predict(X)
