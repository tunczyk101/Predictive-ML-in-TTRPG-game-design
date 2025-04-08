import numpy as np
import pandas as pd
from condor_pytorch import logits_to_label
from sklearn.metrics import accuracy_score
from skorch import NeuralNet
from torch import nn, tensor

from training.constants import NUM_CLASSES


class Condor(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES - 1),
        )

    def forward(self, x):
        logits = self.features(x)
        return logits


class CondorNeuralNet(NeuralNet):
    def fit(self, X, y=None, **fit_params):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)

        super().fit(X, y, **fit_params)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)

        results = super().predict(X)
        return logits_to_label(tensor(results)).numpy()
