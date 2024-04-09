import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from training.constants import DATASET_FILES, FEATURES
from training.create_model import get_fitted_model
from training.creating_dataset import load_and_preprocess_data
from training.splitting_dataset import split_dataframe


def print_results(model_name: str, results: dict):
    print(f"==== {model_name} ====")
    for key, value in results.items():
        print(f"\t--> {key}")
        for measure, m_value in value.items():
            print(f"\t\t--> {measure}: {m_value}")
    print()


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
            "train": {
                "rmse": mean_squared_error(y_train, y_pred_train, squared=False),
                "mse": mean_squared_error(y_train, y_pred_train, squared=True),
                "mae": mean_absolute_error(y_train, y_pred_train),
            },
            "test": {
                "rmse": mean_squared_error(y_test, y_pred_test, squared=False),
                "mse": mean_squared_error(y_test, y_pred_test, squared=True),
                "mae": mean_absolute_error(y_test, y_pred_test),
            },
        }
        results[model_name] = model_results
        if print_summary:
            print_results(model_name, model_results)

    return results
