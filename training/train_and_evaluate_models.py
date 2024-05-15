import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, root_mean_squared_error

from training.create_model import get_fitted_model


def print_results(model_name: str, results: dict):
    print(f"==== {model_name} ====")
    for round_type, rounded_results in results.items():
        print(f"\t--> {round_type}")
        for set_name, value in rounded_results.items():
            print(f"\t\t--> {set_name}")
            for measure, m_value in value.items():
                print(f"\t\t\t--> {measure}: {m_value}")
    print()


def round_results(y_pred, threshold):
    threshold_predict = np.where(
        (y_pred % 1) >= threshold, np.ceil(y_pred), np.floor(y_pred)
    ).astype("int")
    return np.where(threshold_predict > 20, 21, threshold_predict)


def calculate_results(y_true, y_pred, accuracy=True):
    results = {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }
    if accuracy:
        results["accuracy"] = accuracy_score(y_true, y_pred)
    return results


def train_and_evaluate_models(
    models: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    print_summary: bool = True,
) -> dict:
    results = {}

    for model_name in models:
        model = get_fitted_model(model_name, X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        model_results = {
            "no_rounding": {
                "train": calculate_results(y_train, y_pred_train, False),
                "test": calculate_results(y_test, y_pred_test, False),
            },
            "round 0.5": {
                "train": calculate_results(y_train, round_results(y_pred_train, 0.5)),
                "test": calculate_results(y_test, round_results(y_pred_test, 0.5)),
            },
        }

        results[model_name] = model_results
        if print_summary:
            print_results(model_name, model_results)

    return results
