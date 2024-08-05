import os

from dataset.constants import (
    DATASET_FILES,
    FEATURES,
    ORDERED_CHARACTERISTICS_BASIC,
    ORDERED_CHARACTERISTICS_EXPANDED,
    ORDERED_CHARACTERISTICS_FULL,
)
from dataset.creating_dataset import load_and_preprocess_data


current_path = os.getcwd()
DATASET_FILES_PATHS = [
    os.path.join("pathfinder_2e_remaster_data", f) for f in DATASET_FILES
]
BASIC_COLUMNS = ["book", "level"]


if __name__ == "__main__":
    df = load_and_preprocess_data(
        DATASET_FILES_PATHS,
        characteristics=FEATURES,
    )

    df = df[ORDERED_CHARACTERISTICS_FULL + BASIC_COLUMNS]
    df.to_csv("../preprocessed_bestiaries/bestiaries_full.csv")

    df = df[ORDERED_CHARACTERISTICS_EXPANDED + BASIC_COLUMNS]
    df.to_csv("../preprocessed_bestiaries/bestiaries_expanded.csv")

    df = df[ORDERED_CHARACTERISTICS_BASIC + BASIC_COLUMNS]
    df.to_csv("../preprocessed_bestiaries/bestiaries_basic.csv")
