from typing import List

from scipy.stats import somersd


MIN_LEVEL = 0
MAX_LEVEL = 22


def mae_macroaveraged(y_true: List[int], y_predicted: List[int]) -> float:
    """
    Calculates macroaveraged MAE as described in "Evaluation Measures for Ordinal Regression" by S. Baccianella,
    A. Esuli and F. Sebastiani
    :param y_true: Correct levels
    :param y_predicted: Predicted levels
    :return: Value of macroaveraged MAE
    """
    result = 0
    for j in range(MIN_LEVEL, MAX_LEVEL + 1):
        j_class_count = 0
        prediction_result = 0
        for i in range(len(y_predicted)):
            if y_predicted[i] == j:
                j_class_count += 1
                prediction_result += abs(y_true[i] - y_predicted[i])
        result += prediction_result / j_class_count
    return result / (MAX_LEVEL + 1 - MIN_LEVEL)


def mse_macroaveraged(y_true: List[int], y_predicted: List[int]) -> float:
    """
    Calculates macroaveraged MSE as described in "Evaluation Measures for Ordinal Regression" by S. Baccianella,
    A. Esuli and F. Sebastiani
    :param y_true: Correct levels
    :param y_predicted: Predicted levels
    :return: Value of macroaveraged MSE
    """
    result = 0
    for j in range(MIN_LEVEL, MAX_LEVEL + 1):
        j_class_count = 0
        prediction_result = 0
        for i in range(len(y_predicted)):
            if y_predicted[i] == j:
                j_class_count += 1
                prediction_result += (y_true[i] - y_predicted[i]) ** 2
        result += prediction_result / j_class_count
    return result / (MAX_LEVEL + 1 - MIN_LEVEL)


def somers_d(y_true: List[int], y_predicted: List[int]) -> float:
    """
    Calculates Somers' D statistic value.
    :param y_true: Correct levels
    :param y_predicted: Predicted levels
    :return: Value of Somers' D statistic
    """
    somers_d_value = somersd(y_true, y_predicted)
    return somers_d_value.statistic


def accuracy2(y_true: List[int], y_predicted: List[int]) -> float:
    """
    Calculates accuracy of prediction, allowing error of at most 2 classes.
    :param y_true: Correct levels
    :param y_predicted: Predicted levels
    :return: Value of ACC2
    """
    result = 0
    for i in range(len(y_true)):
        if abs(y_true[i] - y_predicted[i]) <= 2:
            result += 1
    return result / len(y_true)

