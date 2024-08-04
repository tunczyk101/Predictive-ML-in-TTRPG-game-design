from dataset.constants import DATASET_FILES, FEATURES
from dataset.creating_dataset import load_and_preprocess_data, min_max_scale_data
from dataset.splitting_dataset import (
    get_dataframe_with_oldest_books,
    load_csv_mapping_file,
)


TEST_RESULT_FILE = "results/results_test_scenarios.csv"
TRAIN_RESULT_FILE = "results/results_train_scenarios.csv"


if __name__ == "__main__":
    bestiaries = load_and_preprocess_data(
        [f"../dataset/pathfinder_2e_remaster_data/{f}" for f in DATASET_FILES],
        FEATURES + ["aoo", "spell_dc"],
    )
    bestiaries = min_max_scale_data(bestiaries)

    all_books = set(load_csv_mapping_file().book)
    training_set = set(get_dataframe_with_oldest_books(bestiaries).book.unique())
    testing_set = all_books - training_set

    print("Training set\n")
    for i in training_set:
        print(i)

    print("\n\ntesting set\n")
    for i in testing_set:
        print(i)

    # X_train, X_test, y_train, y_test = split_dataframe(bestiaries)
    #
    # results_test, results_train = train_and_evaluate_models(
    #     [
    #         "linear_regression",
    #         "linear_regression_ridge",
    #         "linear_regression_lasso",
    #         "lad_regression",
    #         "huber_regression",
    #         "linear_svm",
    #         "kernel_svm",
    #         "knn",
    #         "random_forest",
    #         "lightgbm",
    #         "ordered_model_probit_bfgs",
    #         "ordered_model_logit_bfgs",
    #     ],
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     thresholds=[[0.05 * i for i in range(1, 20)], [0.05 * i for i in range(5, 16)]],
    #     save_files=(TRAIN_RESULT_FILE, TEST_RESULT_FILE)
    # )
