from pprint import pprint

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from training.creating_dataset import load_and_preprocess_data
from training.constants import FEATURES, DATASET_FILES, ORDERED_CHARACTERISTICS_FULL
from training.splitting_dataset import get_date_books_mapping, split_dataframe
from training.train_and_evaluate_models import train_and_evaluate_models

bestiaries = load_and_preprocess_data(
    [f"pathfinder_2e_remaster_data/{f}" for f in DATASET_FILES], FEATURES
)

columns = [col for col in bestiaries.columns if col not in ["book", "level"]]
scaler = MinMaxScaler()
min_max_df = pd.DataFrame()
min_max_df[columns] = pd.DataFrame(
    scaler.fit_transform(bestiaries[columns]), index=bestiaries.index
)
min_max_df["book"] = bestiaries["book"]
min_max_df["level"] = bestiaries["level"]
bestiaries = min_max_df
bestiaries = bestiaries[
    [
        characteristic
        for characteristic in ORDERED_CHARACTERISTICS_FULL
        if characteristic != "aoo"
    ]
    + ["book", "level"]
]

books_dates_map = get_date_books_mapping()

books_to_include = [
    book for _, row in books_dates_map["books"].iteritems() for book in row
]
bestiaries = bestiaries[bestiaries["book"].isin(books_to_include)]
X_train, X_test, y_train, y_test = split_dataframe(bestiaries)

results = train_and_evaluate_models(
    ["linear_ordinal_model"],
    X_train,
    y_train,
    X_test,
    y_test,
    thresholds=[],
    single_threshold=False,
    multiple_thresholds=False,
    graph_thresholds=False,
)
pprint(results)
