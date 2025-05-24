import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


MODEL_LABEL = {
    "baseline": "Baseline",
    "linear_regression": "Linear regression",
    "linear_regression_ridge": "Ridge regression",
    "linear_regression_lasso": "Lasso regression",
    "lad_regression": "LAD regression",
    "huber_regression": "Huber regression",
    "linear_svm": "Linear SVM",
    "kernel_svm": "Kernel SVM",
    "knn": "KNeighbours",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "linear_ordinal_model_probit": "Ordered Model\n[probit]",
    "linear_ordinal_model_logit": "Ordered Model\n[logit]",
    "ordered_random_forest": "Ordered\nRandom Forest",
    "logisticAT": "Logistic AT",
    "logisticIT": "Logistic IT",
    "simple_or": "Simple OR RF",
    "gpor": "GPOR",
    "coral": "CORAL",
    "corn": "CORN",
    "spacecutter": "Spacecutter",
    "nn_rank": "NNRank",
    "condor": "CONDOR",
    "or_cnn": "OR-CNN",
}

MODEL_TO_FUNC = {
    "mae": min,
    "rmse": min,
    "accuracy": max,
}


def find_best_rounding_by_metric(metric: str, model_results: pd.Series) -> str:
    best_rounding = None
    best_result = None

    for rounding_type, metric_name in list(model_results.index):
        if metric == metric_name and rounding_type != "no_rounding":
            metric_val = model_results[(rounding_type, metric_name)]
            if best_result is None or best_result != MODEL_TO_FUNC[metric](
                best_result, metric_val
            ):
                best_rounding = rounding_type
                best_result = metric_val

    return best_rounding


def find_value_for_rounding_metric(
    rounding: str, metric: str, model_results: pd.Series
) -> tuple[list[float], list[str]]:
    results = []
    metrics_names = []
    for rounding_type, metric_name in list(model_results.index):
        if metric in metric_name and rounding_type == rounding:
            results.append(model_results[(rounding_type, metric_name)])
            metrics_names.append(metric_name)
    return results, metrics_names


def plot_results(
    metric_dict: dict[str, list[float]],
    metric_name: str,
    filename: str | None,
    legend_labels: list[str],
    figsize: tuple[float, float] = (15, 8),
):
    labels = list(metric_dict.keys())
    values = list(metric_dict.values())
    num_bars = len(values[0])  # Number of bars per group
    x = np.arange(len(labels))  # Label locations
    width = 0.8 / num_bars  # Bar width

    colors = [mcolors.CSS4_COLORS["yellowgreen"], mcolors.CSS4_COLORS["royalblue"]]

    plt.figure(figsize=figsize)
    for i in range(num_bars):
        bar_values = [v[i] for v in values]  # Extract the i-th value from each list
        plt.bar(
            x + i * width, bar_values, width, label=legend_labels[i], color=colors[i]
        )

    plt.xlabel("Models", fontsize=20)
    plt.ylabel(metric_name.upper(), fontsize=20)
    plt.title(f"Model Results Comparison: {metric_name.upper()}", fontsize=30)
    plt.xticks(x + width * (num_bars - 1) / 2, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plot_best_results(
    based_on_metric: str,
    metric: str,
    results: pd.DataFrame,
    filename: str | None,
):
    best_rounding = {}
    plot_metric_values = {}
    legend_labels = []

    for model in results.index:
        model_results = results.loc[model]
        best_model_rounding = find_best_rounding_by_metric(
            based_on_metric, model_results
        )
        best_rounding[model] = best_model_rounding

        model_best_result, legend_labels = find_value_for_rounding_metric(
            best_model_rounding, metric, model_results
        )
        plot_metric_values[MODEL_LABEL[model]] = model_best_result

    plot_results(plot_metric_values, metric, filename, legend_labels)

    return best_rounding
