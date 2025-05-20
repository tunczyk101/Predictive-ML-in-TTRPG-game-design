import os
import warnings

import pandas as pd

from dataset.creating_dataset import min_max_scale_data
from dataset.splitting_dataset import get_time_series_split_dataframe
from training.train_and_evaluate_models import (
    expanding_window_train_and_evaluate_models,
)


warnings.simplefilter("ignore")

PATH_TO_DATASET = os.path.join(
    "..", "preprocessed_bestiaries", "bestiaries_reduced.csv"
)
MIN_MONSTERS_NUMBER = 100
START_MONSTERS_NUMBER = 213

if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)
    bestiaries = min_max_scale_data(bestiaries)

    ts_dataframes = get_time_series_split_dataframe(
        bestiaries, MIN_MONSTERS_NUMBER, START_MONSTERS_NUMBER
    )
    print(len(ts_dataframes), len(ts_dataframes[0]))

    results_test, results_train = expanding_window_train_and_evaluate_models(
        [
            "baseline",
            # "linear_regression",
            # "linear_regression_ridge",
            # "linear_regression_lasso",
            # "lad_regression",
            # "huber_regression",
            # "linear_svm",
            # "kernel_svm",
            # "knn",
            # "random_forest",
            # "lightgbm",
            # "linear_ordinal_model_probit",
            # "linear_ordinal_model_logit",
            # "ordered_random_forest",
            # "logisticAT",
            # "logisticIT",
            # "simple_or",
            # "gpor",
            # "coral",
            # "corn",
            # "spacecutter",
            # "nn_rank",
            # "condor",
            # "or_cnn",
        ],
        dataframes=ts_dataframes,
        thresholds=[[0.05 * i for i in range(1, 20)], [0.05 * i for i in range(5, 16)]],
    )
