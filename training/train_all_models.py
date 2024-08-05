import os

import pandas as pd

from dataset.creating_dataset import min_max_scale_data
from dataset.splitting_dataset import split_dataframe
from training.train_and_evaluate_models import (
    get_model_results,
    train_and_evaluate_models,
)


PATH_TO_DATASET = os.path.join("..", "preprocessed_bestiaries", "bestiaries_full.csv")
TEST_RESULT_FILE = os.path.join("results", "results_test_scenarios.csv")
TRAIN_RESULT_FILE = os.path.join("results", "results_train_scenarios.csv")


if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET)
    bestiaries = min_max_scale_data(bestiaries)

    X_train, X_test, y_train, y_test = split_dataframe(bestiaries)

    results_test, results_train = train_and_evaluate_models(
        [
            "linear_regression",
            "linear_regression_ridge",
            "linear_regression_lasso",
            "lad_regression",
            "huber_regression",
            "linear_svm",
            "kernel_svm",
            "knn",
            "random_forest",
            "lightgbm",
            "ordered_model_probit_bfgs",
            "ordered_model_logit_bfgs",
        ],
        X_train,
        y_train,
        X_test,
        y_test,
        thresholds=[[0.05 * i for i in range(1, 20)], [0.05 * i for i in range(5, 16)]],
        save_files=(TRAIN_RESULT_FILE, TEST_RESULT_FILE),
    )
