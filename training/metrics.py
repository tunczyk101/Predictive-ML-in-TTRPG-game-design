import math
from collections import defaultdict

import numpy as np
import pandas as pd
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
    return result


def calculate_average_and_std(
    models_results: dict["str", list[list[float | None]]],
    columns: pd.MultiIndex,
    models: list[str],
):
    final_results = defaultdict(list)
    final_columns = [[], []]
    for column in columns:
        final_columns[0] += [column[0], f"{column[1]}_avg"]
        final_columns[1] += [column[1], f"{column[1]}_std"]

    for model, results in models_results.items():
        for i in range(len(results[0])):
            if results[0][i] is None:
                final_results[model] += [None, None]
                continue
            metric_results = np.array([r[i] for r in results])
            final_results[model] += [metric_results.mean(), metric_results.std()]

    return pd.DataFrame(
        data=list(final_results.values()),
        index=models,
        columns=pd.MultiIndex.from_arrays(final_columns, names=columns.names),
    )
