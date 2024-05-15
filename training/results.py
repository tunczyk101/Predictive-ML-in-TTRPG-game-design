import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    mean_absolute_error,
)

from training.train_and_evaluate_models import round_results


COLORS = {
    "linear_regression": mcolors.CSS4_COLORS["papayawhip"],
    "linear_regression_ridge": mcolors.CSS4_COLORS["greenyellow"],
    "linear_regression_lasso": mcolors.CSS4_COLORS["khaki"],
    "lad_regression": mcolors.CSS4_COLORS["yellowgreen"],
    "huber_regression": mcolors.CSS4_COLORS["gold"],
    "linear_svm": mcolors.CSS4_COLORS["paleturquoise"],
    "kernel_svm": mcolors.CSS4_COLORS["deepskyblue"],
    "knn": mcolors.CSS4_COLORS["purple"],
    "random_forest": mcolors.CSS4_COLORS["thistle"],
    "lightgbm": mcolors.CSS4_COLORS["turquoise"],
}

MODEL_LABEL = {
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
}


def get_single_plot_bar(
    results: dict[str, float], measure_type: str, axis: Axes, bar_width: float = 0.1
):
    for i, (model, value) in enumerate(results.items()):
        axis.bar(
            x=i * bar_width,
            height=value,
            width=bar_width,
            color=COLORS[model],
            label=MODEL_LABEL[model],
            linewidth=1,
            edgecolor="black",
        )

    axis.set_ylabel(measure_type, fontweight="bold")
    axis.set_xticklabels([])
    axis.legend(title="Models", loc="upper center", bbox_to_anchor=(0.5, -0.05))


def plot_results(
    results: dict[str, dict[str, float]],
    sets: list[str],
    measure_types: list[str],
    rounding_types,
    figsize=(6, 8),
    hspace=3,
    wspace=1,
):
    models = results.keys()
    rows = len(sets) * len(measure_types)
    columns = len(rounding_types)

    figure, axis = plt.subplots(nrows=rows, ncols=columns, figsize=figsize)
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    for i, set_type in enumerate(sets):
        for r, rounding_type in enumerate(rounding_types):
            for j, measure_type in enumerate(measure_types):
                if measure_type == "accuracy" and rounding_type == "no_rounding":
                    axis[len(measure_types) * i + j][r].axis("off")
                    continue
                measure_results = {
                    m: results[m][rounding_type][set_type][measure_type] for m in models
                }
                get_single_plot_bar(
                    measure_results,
                    measure_type=measure_type,
                    axis=axis[len(measure_types) * i + j][r],
                )
                title = f"[{set_type}]\n {rounding_type}"
                axis[len(measure_types) * i + j][r].set_title(
                    title, fontweight="bold", fontsize=12
                )

    plt.show()


def plot_mae_by_level(
    y_test: pd.Series,
    y_pred_test: np.ndarray,
    title: str = None,
    figsize: tuple[int, int] = (20, 8),
    export: bool = False,
) -> None:
    """
    Plots Mean Absolute Error (MAE) by level.

    Calculates MAE for each level and displays the value on a bar chart.

    :param y_test: True values.
    :param y_pred_test: Predicted values.
    :param title: Plot title.
    :param figsize: A tuple specifying the figure size (width, height). Default is (20, 8).
    :param export: If true, saves plot to results_diagrams file. Default is False.
    :return: None
    """

    y_test = y_test.reset_index(drop=True)
    level_max = y_test.max()

    mae_by_level = pd.DataFrame(columns=["level", "mae"])
    for lvl in range(-1, level_max + 1):
        y_test_curr = y_test[y_test == lvl]
        y_pred_test_curr = pd.DataFrame(y_pred_test)[y_test == lvl]

        mae = mean_absolute_error(y_test_curr, y_pred_test_curr)
        mae_by_level.loc[lvl + 1] = [lvl, mae]

    fig, ax = plt.subplots(figsize=figsize)
    plt.figure(figsize=figsize)
    plt.bar(mae_by_level["level"], mae_by_level["mae"])
    plt.xlabel("Level", fontweight="bold", fontsize=20)
    plt.ylabel("Mean Absolute Error (MAE)", fontweight="bold", fontsize=20)

    if title is None:
        plt.title("MAE by level", fontsize=23, fontweight="bold")
    else:
        plt.title(title, fontsize=23, fontweight="bold")

    plt.xticks(mae_by_level["level"])

    fig.tight_layout()
    if export:
        plt.savefig(f"../results_diagrams/other/mae_by_level/{title}.svg")

    plt.show()


def plot_confusion_matrix(
    predict: np.ndarray,
    y: pd.Series,
    threshold: float = 0.5,
    title: str = None,
    figsize: tuple[int, int] = (10, 10),
    export: bool = False,
) -> None:
    """
    Plots a confusion matrix for rounded predictions based on a specified threshold.
    It visualizes the confusion matrix using a heatmap.

    :param predict: Predicted values to be rounded.
    :param y: True values.
    :param threshold: A round type threshold as a float between 0 and 1. Default is 0.5.
    :param title: Plot title.
    :param figsize: A tuple specifying the figure size (width, height). Default is (10, 10).
    :param export: If true, saves plot to results_diagrams file. Default is False.
    :return: None
    """
    round_predict = round_results(predict, threshold)
    cm = confusion_matrix(y, round_predict)

    # min possible level: -1, max possible level: 21
    labels = [i for i in range(-1, 22)]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, colorbar=False)

    # Adding custom colorbar
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(disp.im_, cax=cax)

    disp.ax_.set_xlabel("Predicted level", fontweight="bold", fontsize=20)
    disp.ax_.set_ylabel("True level", fontweight="bold", fontsize=20)

    if title is None:
        disp.ax_.set_title("Confusion matrix", fontweight="bold", fontsize=20)
    else:
        disp.ax_.set_title(title, fontweight="bold", fontsize=20)

    if export:
        title = title.replace("\n", " ")
        fig.savefig(
            f"../results_diagrams/other/confusion_matrix/{title}.svg",
            bbox_inches="tight",
        )

    plt.show()
