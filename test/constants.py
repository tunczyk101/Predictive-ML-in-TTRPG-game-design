import os
import pathlib


current_path = os.getcwd()
DATASETS_DIR = pathlib.Path(current_path).parent / "pathfinder_2e_remaster_data"
DATASET_FILES = [
    "pathfinder-bestiary.json",
    "pathfinder-bestiary-2.json",
    "pathfinder-bestiary-3.json",
]
DATASET_PATHS = [f"{DATASETS_DIR}/{file}" for file in DATASET_FILES]
