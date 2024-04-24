import json
import os

from training.constants import DATASET_FILES


if __name__ == "__main__":
    base_path = f"../pathfinder_2e_remaster_data"
    original_data_folder = f"{base_path}/packs_json"
    dataset_folders_paths = os.listdir(original_data_folder)
    dataset_folders = [folder[:-5] for folder in DATASET_FILES]
    dataset_folders_paths = [
        folder for folder in dataset_folders_paths if folder in dataset_folders
    ]

    for folder in dataset_folders_paths:
        with open(f"{base_path}/{folder}.json", "a") as bestiary_file:
            for monster in os.listdir(f"{original_data_folder}/{folder}"):
                if "json" in monster and monster.startswith("_"):
                    with open(
                        f"{original_data_folder}/{folder}/{monster}"
                    ) as monster_file:
                        monster_json = json.load(monster_file)
                        json.dump(monster_json, bestiary_file)
                        bestiary_file.write("\n")
