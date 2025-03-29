import numpy as np
import pandas as pd
import torch
from skorch import NeuralNet
from torch import nn

from training.constants import MIN_LVL, NUM_CLASSES


class NNRank(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
            nn.Sigmoid(),
        )

    def forward(self, x) -> torch.Tensor:
        return self.network(x)


class NeuralNetNNRank(NeuralNet):
    def encode_levels(self, y: np.array) -> np.array:
        encoded_y = []
        for i in y:
            encoded_i = np.zeros(NUM_CLASSES)
            encoded_i[: i - MIN_LVL + 1] = 1
            encoded_y.append(encoded_i)
        return np.array(encoded_y, dtype=np.float32)

    def decode_levels(self, y: np.array) -> np.array:
        decoded_y = []
        for encoded_y in y:
            decoded_i = MIN_LVL
            j = 1
            while j < len(encoded_y) and encoded_y[j] >= 0.5:
                decoded_i += 1
                j += 1
            decoded_y.append(decoded_i)
        return np.array(decoded_y)

    def fit(self, X, y=None, **fit_params):
        y_encoded = self.encode_levels(y)
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)

        super().fit(X, y_encoded, **fit_params)
        return self

    def predict(self, X) -> np.array:
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        results = super().predict(X)[:, 1, :]

        return self.decode_levels(results)
