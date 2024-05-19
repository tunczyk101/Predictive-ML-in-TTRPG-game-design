import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, root_mean_squared_error

from training.create_model import get_fitted_model
from training.rounging import (
    find_best_thresholds,
    find_graph_rounding,
    find_single_best_threshold,
    round_results_multiple_threshold,
    round_single_threshold_results,
)


def print_results(model_name: str, results: dict):
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


def calculate_results(y_true, y_pred, accuracy=True):
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
    multiple_thresholds,
    graph_thresholds,
):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

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

    if len(thresholds) > 0:
        best_single_threshold = find_single_best_threshold(
            y_pred_train, y_train, thresholds
        )[0]
        model_results["best_single_threshold"] = {
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
            list(y_train),
            list(y_pred_train),
            thresholds=(min(thresholds), max(thresholds)),
        )
        model_results["best_multiple_thresholds"] = {
            "thresholds": best_thresholds,
            "train": calculate_results(
                y_train, round_results_multiple_threshold(y_pred_train, best_thresholds)
            ),
            "test": calculate_results(
                y_test, round_results_multiple_threshold(y_pred_test, best_thresholds)
            ),
        }

    if graph_thresholds:
        best_thresholds = find_graph_rounding(
            list(y_pred_train), list(y_train), thresholds
        )
        model_results["best_graph_thresholds"] = {
            "thresholds": best_thresholds,
            "train": calculate_results(
                y_train, round_results_multiple_threshold(y_pred_train, best_thresholds)
            ),
            "test": calculate_results(
                y_test, round_results_multiple_threshold(y_pred_test, best_thresholds)
            ),
        }

    return model_results


def train_and_evaluate_models(
    models: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float],
    multiple_thresholds: bool = True,
    graph_thresholds: bool = True,
    print_summary: bool = False,
    return_models: bool = True,
) -> dict:
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
    multiple_thresholds: bool = True,
    graph_thresholds: bool = True,
    print_summary: bool = False,
):
    results = {}

    for model_name, model in models.items():

        results[model_name] = get_model_results(
            model,
            y_train,
            X_train,
            y_test,
            X_test,
            thresholds,
            multiple_thresholds,
            graph_thresholds,
        )

        if print_summary:
            print_results(model_name, results[model_name])

        results[model_name]["model"] = model

    return results
