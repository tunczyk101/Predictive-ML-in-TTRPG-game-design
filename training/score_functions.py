import pandas as pd
from sklearn.metrics import mean_absolute_error


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
