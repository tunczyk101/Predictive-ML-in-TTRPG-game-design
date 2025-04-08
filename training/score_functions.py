import pandas as pd
import torch
from condor_pytorch import condor_negloglikeloss
from coral_pytorch.dataset import levels_from_labelbatch
from sklearn.metrics import mean_absolute_error
from torch import nn

from training.constants import NUM_CLASSES


def orf_mean_absolute_error(y_true, y_pred) -> float:
    """
     Computes the Mean Absolute Error (MAE) between the true labels and the predicted labels for OrderedForest model.

    :param y_true: array-like of shape (n_samples,)
        True labels
    :param y_pred: A dictionary containing the predicted probabilities under the key `"predictions"`.
        The value associated with `"predictions"` should be a pandas DataFrame where each
        row corresponds to a sample and each column corresponds to a class probability.
    :return: Mean Absolute Error
    """
    y_pred = pd.DataFrame(y_pred["predictions"]).idxmax(axis=1)

    return mean_absolute_error(y_true, y_pred)


def spacecutter_mean_absolute_error(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred.argmax(axis=1))


class CondorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.tensor, labels: torch.tensor):
        levels = levels_from_labelbatch(labels, NUM_CLASSES)
        return condor_negloglikeloss(logits, levels)
