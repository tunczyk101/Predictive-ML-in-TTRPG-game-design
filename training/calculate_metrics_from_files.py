import os
import warnings

import pandas as pd

from dataset.creating_dataset import min_max_scale_data
from dataset.splitting_dataset import split_dataframe
from training.train_and_evaluate_models import calculate_results_from_files


warnings.simplefilter("ignore")

PATH_TO_DATASET = os.path.join(
    "..", "preprocessed_bestiaries", "bestiaries_reduced.csv"
)
TEST_RESULT_FILE = os.path.join("results", "results_test.xlsx")
TRAIN_RESULT_FILE = os.path.join("results", "results_train.xlsx")


if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)
    bestiaries = min_max_scale_data(bestiaries)

    X_train, X_test, y_train, y_test = split_dataframe(bestiaries)

    results_test, results_train = calculate_results_from_files(
        [
            "baseline",
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
            "linear_ordinal_model_probit",
            "linear_ordinal_model_logit",
            "ordered_random_forest",
            "logisticAT",
            "logisticIT",
            "simple_or",
            "gpor",
            "coral",
            "corn",
            "spacecutter",
            "nn_rank",
            "condor",
            "or_cnn",
        ],
        y_train,
        y_test,
        thresholds=[[0.05 * i for i in range(1, 20)], [0.05 * i for i in range(5, 16)]],
        save_files=(TRAIN_RESULT_FILE, TEST_RESULT_FILE),
    )
