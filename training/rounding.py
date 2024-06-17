from queue import PriorityQueue

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error


def round_single_threshold_results(
    y_pred: np.ndarray | pd.Series, threshold: float
) -> np.ndarray:
    """
    Rounds predictions based on a specified threshold with maximum rounded value at 21.

    :param y_pred: Predicted values
    :param threshold: Threshold for rounding
    :return: An array of the rounded predictions
    """
    threshold_predict = np.where(
        (y_pred % 1) >= threshold, np.ceil(y_pred), np.floor(y_pred)
    ).astype("int")
    return np.where(threshold_predict > 20, 21, threshold_predict)


def find_single_best_threshold(
    y_pred: np.ndarray | pd.Series, y_true: pd.Series, thresholds: list[float]
) -> float:
    """
    Finds the best threshold for rounding predictions to minimize the mean absolute error (MAE).

    :param y_pred: Predicted values
    :param y_true: True values
    :param thresholds:  A list of threshold values to test, each between 0.0 and 1.0
    :return: The best threshold
    """
    best = (thresholds[0], 21)

    for threshold in thresholds:
        if threshold < 0 or threshold >= 1.0:
            raise ValueError(
                f"Incorrect threshold value. Should be between 0.0 and 1.0 but is {threshold}."
            )

        threshold_predict = round_single_threshold_results(y_pred, threshold)

        mae = mean_absolute_error(y_true, threshold_predict)
        if mae < best[1]:
            best = threshold, mae

    return best[0]


def round_prediction(predicted: float, threshold: float) -> int:
    """
    Rounds a single predicted value based on a specified threshold.
    Minimum possible rounded value is -1 and maximum possible rounded value is 21.

    :param predicted: Predicted value
    :param threshold: Threshold for rounding
    :return: Rounded value
    """
    if threshold is None:
        return min(21, max(-1, predicted))

    round_val = predicted // 1

    if predicted % 1 >= threshold:
        round_val += 1
    return round_val


def round_prediction_error(predicted: float, true: float, threshold: float) -> float:
    """
     Calculates the absolute error between the true value and the rounded predicted value based on a specified threshold.

    :param predicted: Predicted value
    :param true: True value
    :param threshold: Threshold for rounding
    :return: The absolute error between the true value and the rounded predicted value.
    """
    return abs(true - round_prediction(predicted, threshold))


def round_results_multiple_threshold(
    y_predicted: np.ndarray, thresholds: dict[int, float]
) -> list[int]:
    """
    Rounds a list of predicted values based on multiple thresholds specified for each integer part of the prediction.

    :param y_predicted: Predicted values to be rounded
    :param thresholds: Dictionary with thresholds
    :return: A list of rounded values.
    """
    return [
        round_prediction(prediction, thresholds.get(prediction // 1))
        for prediction in y_predicted
    ]


def objective(
    trial: optuna.trial.Trial,
    y_true: list[int],
    y_predicted: list[float],
    thresholds: tuple[float, float],
) -> float:
    """
    Objective function for optimizing thresholds to minimize the mean absolute error.

    :param trial: Optimization trial object used to suggest threshold values
    :param y_true: List of true values
    :param y_predicted: List of predicted values
    :param thresholds: Tuple containing the lower and upper bounds for the thresholds
    :return: The mean absolute error (MAE) between the true values and the rounded predicted values
    """
    level_thresholds = {
        i: trial.suggest_float(f"level_{i}", thresholds[0], thresholds[1])
        for i in range(-1, 21)
    }
    n = len(y_true)
    sum_prediction_error = sum(
        [
            round_prediction_error(
                y_predicted[i], y_true[i], level_thresholds.get(y_predicted[i] // 1)
            )
            for i in range(n)
        ]
    )
    mean_prediction_error = sum_prediction_error / n
    return mean_prediction_error


def find_best_thresholds(
    y_predicted: list[float],
    y_true: list[int],
    thresholds: tuple[float, float] = (0, 1),
) -> dict[int, float]:
    """
    Finds the best thresholds for rounding predicted values to minimize the mean absolute error (MAE).

    :param y_predicted: List of predicted values
    :param y_true: List of true values
    :param thresholds: Tuple containing the lower and upper bounds for the thresholds
    :return: Dictionary mapping level to their optimized thresholds.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, y_true, y_predicted, thresholds), n_trials=100
    )
    return {i - 1: threshold for i, threshold in enumerate(study.best_params.values())}


def get_edges_cost(
    level: int, thresholds: list[float], y_pred: list[float], y_true: list[int]
) -> list[tuple[float, float]]:
    """
    Calculates the cost for each threshold at a given level.

    :param level: Level - integer part of the predicted values to consider
    :param thresholds: Tuple containing the lower and upper bounds for the thresholds
    :param y_pred: List of predicted values
    :param y_true: List of true values
    :return: List of tuples representing each edge (threshold) from this edge and corresponding cost of this edge (MAE).
    """
    lvl_pred = [pred for pred in range(len(y_pred)) if y_pred[pred] // 1 == level]
    n = len(lvl_pred)
    moves = []
    for threshold in thresholds:
        sum_prediction_error = sum(
            [round_prediction_error(y_pred[i], y_true[i], threshold) for i in lvl_pred]
        )
        mean_prediction_error = sum_prediction_error / n
        moves.append((threshold, mean_prediction_error))

    return moves


def find_graph_rounding(
    y_pred: list[float], y_true: list[int], thresholds: list[float]
) -> dict[int, float]:
    """
     Finds the best thresholds for rounding using a graph-based approach.

    :param y_pred: List of predicted values
    :param y_true: List of true values
    :param thresholds: List of threshold values to consider for rounding
    :return: Dictionary mapping level to their optimized thresholds
    """
    q = PriorityQueue()
    final_thresholds = {}
    q.put((0, -1))

    while not q.empty():
        cost, level = q.get()

        if level == 21:
            return final_thresholds

        edges_cost = get_edges_cost(level, thresholds, y_pred, y_true)
        threshold, next_edge_cost = min(edges_cost, key=lambda x: x[1])
        final_thresholds[level] = threshold
        q.put((cost + next_edge_cost, level + 1))
