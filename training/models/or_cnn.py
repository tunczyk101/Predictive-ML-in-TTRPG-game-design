import numpy as np
import pandas as pd
from skorch import NeuralNet
from torch import nn, tensor

from training.constants import MIN_LVL, NUM_CLASSES
from training.models.condor import Condor
from training.models.nn_rank import NeuralNetNNRank


class ORCNN(Condor):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self.features = nn.Sequential(
            self.features,
            nn.Sigmoid(),
        )


class NeuralNetORCNN(NeuralNetNNRank):
    def encode_levels(self, y: np.ndarray) -> np.ndarray:
        encoded_y = []
        for i in y:
            encoded_i = np.zeros(NUM_CLASSES - 1)
            encoded_i[: i - MIN_LVL] = 1
            encoded_y.append(encoded_i)
        return np.array(encoded_y, dtype=np.float32)

    def decode_levels(self, y: np.array) -> np.array:
        decoded_y = []
        for encoded_y in y:
            decoded_i = MIN_LVL
            j = 0
            while j < len(encoded_y) and encoded_y[j] >= 0.5:
                decoded_i += 1
                j += 1
            decoded_y.append(decoded_i)
        return np.array(decoded_y)

    def predict(self, X) -> np.array:
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        results = NeuralNet.predict(self, X=X)

        return self.decode_levels(results)
