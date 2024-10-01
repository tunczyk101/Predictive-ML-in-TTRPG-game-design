import gpflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class GPOR(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.cutpoints = np.linspace(-10, 10, 22)
        self.likelihood = gpflow.likelihoods.Ordinal(self.cutpoints)
        self.kernel = gpflow.kernels.SquaredExponential()

    def fit(self, X, y):
        model = gpflow.models.VGP(
            (X, pd.DataFrame(y)), kernel=self.kernel, likelihood=self.likelihood
        )
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(
            model.training_loss, model.trainable_variables, options=dict(maxiter=100)
        )
        self.model_ = model

    def predict(self, X):
        f_mean, _ = self.model_.predict_y(np.array(X))
        return f_mean.numpy().flatten()
