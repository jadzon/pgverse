import os
import glob


def create_yolo_dataset(dataset_dir="../dataset/ele_dataset_prepared_aug"):

    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_dict = {class_name: idx for idx, class_name in enumerate(sorted(class_dirs))}

    for class_name, class_id in class_dict.items():
        class_path = os.path.join(dataset_dir, class_name)
        image_files = glob.glob(os.path.join(class_path, "*.png"))

        for img_path in image_files:
            center_x = 0.5
            center_y = 0.5
            box_width = 0.8
            box_height = 0.8

            label_content = f"{class_id} {center_x} {center_y} {box_width} {box_height}"

            label_path = os.path.splitext(img_path)[0] + ".txt"
            with open(label_path, "w") as f:
                f.write(label_content)

    # Utwórz plik data.yaml
    yaml_content = f"""
# YOLOv8 Dataset Configuration
path: ./dataset  # Ścieżka do folderu z danymi
train: ele_dataset_prepared  # Ścieżka względna do folderu treningowego

# Classes
nc: {len(class_dict)}  # Liczba klas
names: {list(class_dict.keys())}  # Nazwy klas
    """

    # Zapisz plik data.yaml
    yaml_path = os.path.join("../dataset", "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(
        f"Utworzono etykiety dla {sum(len(glob.glob(os.path.join(dataset_dir, class_name, '*.png'))) for class_name in class_dict)} obrazów.")
    print(f"Utworzono plik konfiguracyjny data.yaml w {yaml_path}")
    print(f"Wykryto {len(class_dict)} klas: {', '.join(class_dict.keys())}")


if __name__ == "__main__":
    create_yolo_dataset()