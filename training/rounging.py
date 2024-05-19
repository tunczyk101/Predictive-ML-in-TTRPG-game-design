from queue import PriorityQueue

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error


def round_single_threshold_results(y_pred, threshold):
    threshold_predict = np.where(
        (y_pred % 1) >= threshold, np.ceil(y_pred), np.floor(y_pred)
    ).astype("int")
    return np.where(threshold_predict > 20, 21, threshold_predict)


def find_single_best_threshold(
    y_pred: pd.Series, y_true: pd.Series, thresholds: list[float]
):
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

    return best


def round_prediction(predicted, threshold):
    if threshold is None:
        return min(21, max(-1, predicted))

    round_val = predicted // 1

    if predicted % 1 >= threshold:
        round_val += 1
    return round_val


def round_prediction_error(predicted, true, threshold):
    return abs(true - round_prediction(predicted, threshold))


def round_results_multiple_threshold(y_predicted, thresholds):
    return [
        round_prediction(prediction, thresholds.get(prediction // 1))
        for prediction in y_predicted
    ]


def objective(trial, y_true, y_predicted, thresholds):
    level_thresholds = {
        i: trial.suggest_float(f"level_{i}", thresholds[0], thresholds[1])
        for i in range(-1, 21)
    }
    n = len(y_true)
    return (
        sum(
            [
                round_prediction_error(
                    y_predicted[i], y_true[i], level_thresholds.get(y_predicted[i] // 1)
                )
                for i in range(n)
            ]
        )
        / n
    )


def find_best_thresholds(y_true, y_predicted, thresholds=(0, 1)):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, y_true, y_predicted, thresholds), n_trials=100
    )
    return {i - 1: threshold for i, threshold in enumerate(study.best_params.values())}


def get_edges_cost(level, thresholds, y_pred, y_true):
    lvl_pred = [pred for pred in range(len(y_pred)) if y_pred[pred] // 1 == level]
    n = len(lvl_pred)
    moves = []
    for threshold in thresholds:
        moves.append(
            (
                threshold,
                sum(
                    [
                        round_prediction_error(y_pred[i], y_true[i], threshold)
                        for i in lvl_pred
                    ]
                )
                / n,
            )
        )

    return moves


def find_graph_rounding(y_pred, y_true, thresholds):
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
