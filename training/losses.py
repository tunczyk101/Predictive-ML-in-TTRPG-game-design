import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from condor_pytorch import condor_negloglikeloss
from coral_pytorch.dataset import levels_from_labelbatch
from torch import nn

from training.constants import NUM_CLASSES


class CondorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.tensor, labels: torch.tensor):
        levels = levels_from_labelbatch(labels, NUM_CLASSES)
        return condor_negloglikeloss(logits, levels)


class WeightedBCELoss(nn.Module):
    def __init__(self, y_train: np.ndarray | None):
        super().__init__()
        if y_train is None:
            weights = np.ones(NUM_CLASSES - 1)
        else:
            thresholds = np.arange(0, NUM_CLASSES - 1)
            n_t = [np.sum(y_train == t) for t in thresholds]
            sqrt_n = np.sqrt(n_t)
            weights = sqrt_n / np.sum(sqrt_n)

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.tensor, labels: torch.tensor):
        bce = F.binary_cross_entropy(logits, labels, reduction="none")
        weighted_bce = bce * self.weights
        return weighted_bce.mean()
