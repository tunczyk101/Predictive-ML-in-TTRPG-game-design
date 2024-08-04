import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score, mean_absolute_error, root_mean_squared_error
from statsmodels.miscmodels.ordinal_model import OrderedResultsWrapper

from training.create_model import get_fitted_model
from training.rounding import (
    find_best_thresholds,
    find_graph_rounding,
    find_single_best_threshold,
    round_results_multiple_threshold,
    round_single_threshold_results,
)


def calculate_results(y_true, y_pred, include_accuracy=True) -> list[float]:
    """
    Calculates evaluation metrics for predicted values compared to true values.

    :param y_true: List of true values
    :param y_pred: List of predicted values
    :param include_accuracy: Whether to compute the accuracy score.
    :return: List containing evaluation metrics. If accuracy was not computed the last element equals None.
    """
    results = [
        root_mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        None,
    ]
    if include_accuracy:
        results[-1] = accuracy_score(y_true, y_pred)
    return results


def get_index(thresholds: list[tuple[float, float]]):
    """
    Create a pandas MultiIndex based on provided thresholds.

    :param thresholds: A list of tuples, where each tuple contains two float values representing the start
                        and end of a threshold range.
    :return: A MultiIndex object with headers for no rounded results, classic (0.5) rounding
                and for each given threshold pair round for single_threshold, multiple_threshold, graph_threshold.
                For each rounding type there is returned a group metrices.
    """
    iterables = [
        ["no_rounding", "round 0.5"]
        + [
            f"{label}{threshold_start:.2f}_{threshold_end:.2f}"
            for threshold_start, threshold_end in thresholds
            for label in [
                "best_single_threshold_",
                "best_multiple_thresholds_",
                "best_graph_thresholds_",
            ]
        ],
        ["rmse", "mae", "accuracy"],
    ]
    return pd.MultiIndex.from_product(
        iterables, names=["round type + metrics", "model"]
    )


def get_model_results(
    model,
    y_train,
    X_train,
    y_test,
    X_test,
    thresholds,
) -> (list[float], list[float]):
    """
    Calculates and compares evaluation metrics for different rounding strategies based on a machine learning model.

    :param model: Model to evaluate
    :param y_train: True target values for the training set
    :param X_train: Feature matrix for the training set
    :param y_test: True target values for the test set
    :param X_test: Feature matrix for the test set
    :param thresholds: List of threshold values to consider for rounding=
    :return: Two lists containing evaluation metrics for different rounding strategies
    """
    if type(model) != OrderedResultsWrapper:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    else:
        y_pred_train = (
            model.predict(X_train)
            .rename(columns={i + 1: i for i in range(-1, 22)})
            .idxmax(axis=1)
        ) - 1
        y_pred_test = (
            model.predict(X_test)
            .rename(columns={i + 1: i for i in range(-1, 22)})
            .idxmax(axis=1)
        ) - 1

    train_results = calculate_results(
        y_train, y_pred_train, include_accuracy=False
    ) + calculate_results(
        y_train, round_single_threshold_results(y_pred_train, threshold=0.5)
    )

    test_results = calculate_results(
        y_test, y_pred_test, include_accuracy=False
    ) + calculate_results(
        y_test, round_single_threshold_results(y_pred_test, threshold=0.5)
    )

    for threshold_list in thresholds:
        min_threshold = min(threshold_list)
        max_threshold = max(threshold_list)

        best_single_threshold = find_single_best_threshold(
            y_pred_train, y_train, threshold_list
        )

        train_results_single_threshold = calculate_results(
            y_train,
            round_single_threshold_results(y_pred_train, best_single_threshold),
        )
        test_results_single_threshold = calculate_results(
            y_test,
            round_single_threshold_results(y_pred_test, best_single_threshold),
        )

        train_results += train_results_single_threshold
        test_results += test_results_single_threshold

        best_thresholds = find_best_thresholds(
            list(y_pred_train),
            list(y_train),
            thresholds=(min_threshold, max_threshold),
        )

        train_results_multiple_threshold = calculate_results(
            y_train, round_results_multiple_threshold(y_pred_train, best_thresholds)
        )
        test_results_multiple_threshold = calculate_results(
            y_test, round_results_multiple_threshold(y_pred_test, best_thresholds)
        )

        train_results += train_results_multiple_threshold
        test_results += test_results_multiple_threshold

        best_thresholds = find_graph_rounding(
            list(y_pred_train), list(y_train), threshold_list
        )

        train_results_graph = calculate_results(
            y_train, round_results_multiple_threshold(y_pred_train, best_thresholds)
        )
        test_results_graph = calculate_results(
            y_test, round_results_multiple_threshold(y_pred_test, best_thresholds)
        )

        train_results += train_results_graph
        test_results += test_results_graph

    return train_results, test_results


def train_and_evaluate_models(
    models: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[list[float]],
    save_files: tuple[str, str],
) -> tuple[DataFrame, DataFrame]:
    """
    Trains and evaluates multiple machine learning models and compares different rounding strategies.

    :param models: List of model names to train and evaluate
    :param X_train: Feature matrix for the training set
    :param y_train: True target values for the training set
    :param X_test: Feature matrix for the test set
    :param y_test: True target values for the test set
    :param thresholds: List of threshold values to consider for rounding
    :return: Pandas DataFrames containing evaluation metrics for each model and rounding strategy.
                One for test results and another one for train results.
    """
    all_train_results = []
    all_test_results = []
    train_results_file, test_results_file = save_files
    columns = get_index(thresholds=[(min(th), max(th)) for th in thresholds])

    for i, model_name in enumerate(models):
        model = get_fitted_model(model_name, X_train, y_train)
        model_train_results, model_test_results = get_model_results(
            model,
            y_train,
            X_train,
            y_test,
            X_test,
            thresholds,
        )

        all_train_results.append(model_train_results)
        all_test_results.append(model_test_results)

        columns = get_index(thresholds=[(min(th), max(th)) for th in thresholds])
        pd.DataFrame(
            data=all_train_results, index=models[: i + 1], columns=columns
        ).to_csv(train_results_file)
        pd.DataFrame(
            data=all_test_results, index=models[: i + 1], columns=columns
        ).to_csv(test_results_file)

    return pd.DataFrame(
        data=all_test_results, index=models, columns=columns
    ), pd.DataFrame(data=all_train_results, index=models, columns=columns)
