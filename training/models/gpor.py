import gpflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class GPOR(BaseEstimator, RegressorMixin):
    def __init__(self, maxiter=100, kernel=gpflow.kernels.RBF()):
        self.cutpoints = np.linspace(-10, 10, 22)
        self.likelihood = gpflow.likelihoods.Ordinal(self.cutpoints)
        self.kernel = kernel
        self.maxiter = maxiter

    def fit(self, X, y):
        model = gpflow.models.VGP(
            (X, pd.DataFrame(y)), kernel=self.kernel, likelihood=self.likelihood
        )
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(
            model.training_loss,
            model.trainable_variables,
            options=dict(maxiter=self.maxiter),
        )
        self.model_ = model

    def predict(self, X):
        f_mean, _ = self.model_.predict_y(np.array(X))
        return f_mean.numpy().flatten()
