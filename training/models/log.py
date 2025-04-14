import numpy as np
import pandas as pd
import torch
from skorch import NeuralNet
from torch import nn

from training.constants import MIN_LVL, NUM_CLASSES


class LogLossModule(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)


class LogLossNeuralNet(NeuralNet):
    def fit(self, X, y=None, **fit_params):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)

        super().fit(X, y, **fit_params)
        return self

    def predict(self, X) -> np.array:
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        results = super().predict(X)

        return results.argmax(axis=1)
