from typing import Optional

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
    def __init__(self, offset: Optional[float] = None, distr: str = "logit"):
        self.offset = offset
        self.distr = distr

    def fit(self, X, y=None):
        model = RegularizedOrderedModel(
            exog=X, endog=y, distr=self.distr, offset=self.offset
        )
        self.model_ = model.fit(method="bfgs", disp=0)

        return self

    def predict(self, X):
        return self.model_.predict(X).idxmax(axis=1)
