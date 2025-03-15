import os.path
from pathlib import Path

import pandas as pd

from training.ploting import plot_best_results


TEST_RESULT_FILE = os.path.join("results", "results_test.xlsx")
TRAIN_RESULT_FILE = os.path.join("results", "results_train.xlsx")
METRICS_LIST = ["mae", "rmse", "accuracy", "somers_d"]
PLOTS_RESULTS_FILE = os.path.join("results", "plot")
Path(PLOTS_RESULTS_FILE).mkdir(parents=True, exist_ok=True)


def plot_and_save(df: pd.DataFrame, metrics_list: list[str], base_metric: str) -> None:
    #  base_metric - metric used to find the best rounding type for model
    for metric in metrics_list:
        plot_best_results(
            based_on_metric=base_metric,
            metric=metric,
            results=df,
            filename=os.path.join(
                PLOTS_RESULTS_FILE, f"{metric}_based_on_{base_metric}.svg"
            ),
        )


if __name__ == "__main__":
    df_test = pd.read_excel(
        TEST_RESULT_FILE,
        header=[0, 1],
        index_col=[0],
    )
    plot_and_save(df_test, METRICS_LIST, "mae")

    df_train = pd.read_excel(
        TRAIN_RESULT_FILE,
        header=[0, 1],
        index_col=[0],
    )
