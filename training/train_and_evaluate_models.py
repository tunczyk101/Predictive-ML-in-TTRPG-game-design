import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, root_mean_squared_error
from statsmodels.miscmodels.ordinal_model import OrderedResultsWrapper

from training.create_model import get_fitted_model
from training.rounging import (
    find_best_thresholds,
    find_graph_rounding,
    find_single_best_threshold,
    round_results_multiple_threshold,
    round_single_threshold_results,
)


def print_results(model_name: str, results: dict):
    """
    Prints results and model name summary.

    :param model_name: Model name
    :param results: Results of a given model
    """
    print(f"==== {model_name} ====")
    for round_type, rounded_results in results.items():
        if round_type == "model":
            continue
        print(f"\t--> {round_type}")
        for set_name, value in rounded_results.items():
            print(f"\t\t--> {set_name}")
            if set_name == "threshold":
                print(f"\t\t\t--> {value}")
                continue
            for measure, m_value in value.items():
                print(f"\t\t\t--> {measure}: {m_value}")
    print()


def calculate_results(y_true, y_pred, accuracy=True) -> dict[str, float]:
    """
    Calculates evaluation metrics for predicted values compared to true values.

    :param y_true: List of true values
    :param y_pred: List of predicted values
    :param accuracy: Whether to compute the accuracy score
    :return: Dictionary containing evaluation metrics
    """
    results = {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }
    if accuracy:
        results["accuracy"] = accuracy_score(y_true, y_pred)
    return results


def get_model_results(
    model,
    y_train,
    X_train,
    y_test,
    X_test,
    thresholds,
    single_threshold,
    multiple_thresholds,
    graph_thresholds,
) -> dict:
    """
    Calculates and compares evaluation metrics for different rounding strategies based on a machine learning model.

    :param model: Model to evaluate
    :param y_train: True target values for the training set
    :param X_train: Feature matrix for the training set
    :param y_test: True target values for the test set
    :param X_test: Feature matrix for the test set
    :param thresholds: List of threshold values to consider for rounding
    :param single_threshold: Whether to find the single best threshold
    :param multiple_thresholds: Whether to find the multiple thresholds
    :param graph_thresholds: Whether to find the graph-based threshold
    :return: Dictionary containing evaluation metrics for different rounding strategies
    """
    if type(model) != OrderedResultsWrapper:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    else:
        y_pred_train = (
            model.predict(X_train)
            .rename(columns={i + 1: i for i in range(-1, 22)})
            .idxmax(axis=1)
        )
        y_pred_test = (
            model.predict(X_test)
            .rename(columns={i + 1: i for i in range(-1, 22)})
            .idxmax(axis=1)
        )

    model_results = {
        "no_rounding": {
            "train": calculate_results(y_train, y_pred_train, False),
            "test": calculate_results(y_test, y_pred_test, False),
        },
        "round 0.5": {
            "train": calculate_results(
                y_train, round_single_threshold_results(y_pred_train, 0.5)
            ),
            "test": calculate_results(
                y_test, round_single_threshold_results(y_pred_test, 0.5)
            ),
        },
    }

    for threshold_list in thresholds:
        min_threshold = min(threshold_list)
        max_threshold = max(threshold_list)

        if single_threshold:
            best_single_threshold = find_single_best_threshold(
                y_pred_train, y_train, threshold_list
            )[0]
            model_results[
                f"best_single_threshold_{min_threshold:.2f}_{max_threshold:.2f}"
            ] = {
                "threshold": best_single_threshold,
                "train": calculate_results(
                    y_train,
                    round_single_threshold_results(y_pred_train, best_single_threshold),
                ),
                "test": calculate_results(
                    y_test,
                    round_single_threshold_results(y_pred_test, best_single_threshold),
                ),
            }

        if multiple_thresholds:
            best_thresholds = find_best_thresholds(
                list(y_pred_train),
                list(y_train),
                thresholds=(min_threshold, max_threshold),
            )
            model_results[
                f"best_multiple_thresholds_{min_threshold:.2f}_{max_threshold:.2f}"
            ] = {
                "thresholds": best_thresholds,
                "train": calculate_results(
                    y_train,
                    round_results_multiple_threshold(y_pred_train, best_thresholds),
                ),
                "test": calculate_results(
                    y_test,
                    round_results_multiple_threshold(y_pred_test, best_thresholds),
                ),
            }

        if graph_thresholds:
            best_thresholds = find_graph_rounding(
                list(y_pred_train), list(y_train), threshold_list
            )
            model_results[
                f"best_graph_thresholds_{min_threshold:.2f}_{max_threshold:.2f}"
            ] = {
                "thresholds": best_thresholds,
                "train": calculate_results(
                    y_train,
                    round_results_multiple_threshold(y_pred_train, best_thresholds),
                ),
                "test": calculate_results(
                    y_test,
                    round_results_multiple_threshold(y_pred_test, best_thresholds),
                ),
            }

    return model_results


def train_and_evaluate_models(
    models: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[list[float]],
    single_threshold: bool = True,
    multiple_thresholds: bool = True,
    graph_thresholds: bool = True,
    print_summary: bool = False,
    return_models: bool = True,
) -> dict:
    """
    Trains and evaluates multiple machine learning models and compares different rounding strategies.

    :param models: List of model names to train and evaluate
    :param X_train: Feature matrix for the training set
    :param y_train: True target values for the training set
    :param X_test: Feature matrix for the test set
    :param y_test: True target values for the test set
    :param thresholds: List of threshold values to consider for rounding
    :param single_threshold: Whether to find the single best threshold
    :param multiple_thresholds: Whether to find the multiple thresholds
    :param graph_thresholds: Whether to find the graph-based threshold
    :param print_summary: Whether to print summary results for each model
    :param return_models: Whether to return trained models in the results
    :return: Dictionary containing evaluation metrics for each model and rounding strategy
    """
    results = {}

    for model_name in models:
        model = get_fitted_model(model_name, X_train, y_train)

        results[model_name] = get_model_results(
            model,
            y_train,
            X_train,
            y_test,
            X_test,
            thresholds,
            single_threshold,
            multiple_thresholds,
            graph_thresholds,
        )

        if return_models:
            results[model_name]["model"] = model

        if print_summary:
            print_results(model_name, results[model_name])

    return results


def evaluate_models(
    models: dict[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float],
    single_threshold: bool = True,
    multiple_thresholds: bool = True,
    graph_thresholds: bool = True,
    print_summary: bool = False,
) -> dict:
    """
    Evaluates multiple models using different rounding strategies.

    :param models: Dictionary mapping model names to trained models
    :param X_train: Feature matrix for the training set
    :param y_train: True target values for the training set
    :param X_test: Feature matrix for the test set
    :param y_test: True target values for the test set
    :param thresholds: List of threshold values to consider for rounding
    :param single_threshold: Whether to find the single best threshold
    :param multiple_thresholds: Whether to find the multiple thresholds
    :param graph_thresholds: Whether to find the graph-based threshold
    :param print_summary: Whether to print summary results for each model
    :return: Dictionary containing evaluation metrics for each model and rounding strategy
    """
    results = {}

    for model_name, model in models.items():

        results[model_name] = get_model_results(
            model,
            y_train,
            X_train,
            y_test,
            X_test,
            thresholds,
            single_threshold,
            multiple_thresholds,
            graph_thresholds,
        )

        if print_summary:
            print_results(model_name, results[model_name])

        results[model_name]["model"] = model

    return results
