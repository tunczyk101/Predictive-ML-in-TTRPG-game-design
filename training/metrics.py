import math

import numpy as np
from scipy.stats import somersd


MIN_LEVEL = 0
MAX_LEVEL = 22


def mae_macroaveraged(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculates macroaveraged MAE as described in "Evaluation Measures for Ordinal Regression" by S. Baccianella,
    A. Esuli and F. Sebastiani"""
    lvl_counts = np.zeros(MAX_LEVEL - MIN_LEVEL + 1)
    sum_err_per_lvl = np.zeros(MAX_LEVEL - MIN_LEVEL + 1)
    for i, true_lvl in enumerate(y_true):
        lvl_counts[true_lvl - MIN_LEVEL] += 1
        sum_err_per_lvl[true_lvl - MIN_LEVEL] += abs(true_lvl - y_predicted[i])

    avg_err_per_lvl = sum_err_per_lvl[lvl_counts != 0] / lvl_counts[lvl_counts != 0]
    return np.sum(avg_err_per_lvl) / (MAX_LEVEL - MIN_LEVEL + 1)


def mse_macroaveraged(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculates macroaveraged MSE as described in "Evaluation Measures for Ordinal Regression" by S. Baccianella,
    A. Esuli and F. Sebastiani"""
    lvl_counts = np.zeros(MAX_LEVEL - MIN_LEVEL + 1)
    sum_err_per_lvl = np.zeros(MAX_LEVEL - MIN_LEVEL + 1)
    for i, true_lvl in enumerate(y_true):
        lvl_counts[true_lvl - MIN_LEVEL] += 1
        sum_err_per_lvl[true_lvl - MIN_LEVEL] += (true_lvl - y_predicted[i]) ** 2

    avg_err_per_lvl = sum_err_per_lvl[lvl_counts != 0] / lvl_counts[lvl_counts != 0]
    return np.sum(avg_err_per_lvl) / (MAX_LEVEL - MIN_LEVEL + 1)


def rmse_macroaveraged(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculates macroaveraged RMSE as described in "Evaluation Measures for Ordinal Regression" by S. Baccianella,
    A. Esuli and F. Sebastiani"""
    return math.sqrt(mse_macroaveraged(y_true, y_predicted))


def somers_d(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculates Somers' D statistic value."""
    somers_d_value = somersd(y_true, y_predicted)
    return somers_d_value.statistic


def accuracy_at_k(y_true: np.ndarray, y_predicted: np.ndarray, k: int = 0) -> float:
    """Calculates accuracy of prediction, allowing error of at most `k` classes."""
    result = np.sum(np.abs(y_true - y_predicted) <= k) / len(y_true)
    return result[0]
