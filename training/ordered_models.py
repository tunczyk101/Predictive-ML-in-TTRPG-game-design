import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from statsmodels.miscmodels.ordinal_model import OrderedModel


class RegularizedOrderedModel(OrderedModel):
    def __init__(self, alpha: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def loglike(self, params):
        log_likelihoods = super().loglikeobs(params)
        n = len(log_likelihoods)
        l2 = np.sum(np.square(params))
        return log_likelihoods.sum() - self.alpha * n * l2


class LinearOrdinalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha

    def fit(self, X, y=None):
        self.model_ = RegularizedOrderedModel(
            exog=X, endog=y, distr="logit", alpha=self.alpha
        )
        self.model_.fit(method="lbfgs")

    def predict(self, X):
        return (
            self.model_.predict(X)
            .rename(columns={i + 1: i for i in range(-1, 22)})
            .idxmax(axis=1)
        )
