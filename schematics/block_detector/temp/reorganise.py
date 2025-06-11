import os
import shutil
import re
from pathlib import Path
from collections import defaultdict


def reorganize_images(source_folder="../dataset/ELE_DATASET", output_folder="../dataset/ele_dataset_prepared"):
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    file_counters = defaultdict(int)

    for dirpath, dirnames, filenames in os.walk(source_folder):
        current_folder = os.path.basename(dirpath)

        category = get_category(current_folder)

        if not category:
            continue

        category_path = output_path / category
        category_path.mkdir(exist_ok=True)

        for filename in filenames:
            if filename.lower().endswith('.png'):
                source_file = os.path.join(dirpath, filename)

                file_counters[category] += 1

                _, ext = os.path.splitext(filename)
                new_filename = f"{file_counters[category]}{ext}"

                target_file = category_path / new_filename

                shutil.copy2(source_file, target_file)

    remove_empty_folders(output_path)


def get_category(folder_name):
    if folder_name.startswith("MOSFET_N"):
        return "mosfet_n"
    elif folder_name.startswith("MOSFET_P"):
        return "mosfet_p"

    if folder_name in ["circle_npn", "npn"]:
        return "bjt_npn"
    elif folder_name in ["circle_pnp", "pnp"]:
        return "bjt_pnp"

    patterns = {
        r"^gnd_\d+$": "gnd",
        r"^dc_volt_src_\d+$": "dc_volt_src",
        r"^resistor_\d+$": "resistor"
    }

    for pattern, category in patterns.items():
        if re.match(pattern, folder_name):
            return category

    return folder_name


def remove_empty_folders(path):
    for folder in path.iterdir():
        if folder.is_dir():
            remove_empty_folders(folder)

            if not any(folder.iterdir()):
                folder.rmdir()


if __name__ == "__main__":
    reorganize_images()
    print("Zakończono reorganizację zdjęć!")