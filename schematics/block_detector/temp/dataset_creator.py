import os
import re
import cv2
import glob
import random
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image


def reorganize_images(source_folder="./dataset/ELE_DATASET", output_folder="./dataset/ele_dataset_prepared"):
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
    return output_path


def get_category(folder_name):
    if folder_name.startswith("MOSFET_N"):
        return "MOSFET_N"
    elif folder_name.startswith("MOSFET_P"):
        return "MOSFET_P"

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


def augment_images(input_dir, output_dir, augmentations_per_image=12):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Funkcja do augmentacji pojedynczego obrazu
    def augment_image(img):
        augmented_images = []
        h, w = img.shape[:2]

        for i in range(augmentations_per_image):
            aug_img = img.copy()

            # Rotacja (losowy kąt lub wielokrotność 90°)
            angle = (360 / augmentations_per_image) * i
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # Zmiana jasności/kontrastu
            alpha = random.uniform(0.8, 1.2)  # Kontrast
            beta = random.randint(-15, 15)  # Jasność
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

            # Dodanie rozmycia (opcjonalnie)
            if random.random() > 0.7:
                kernel_size = random.choice([3, 5])
                aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)

            augmented_images.append(aug_img)

        return augmented_images

    # Rekurencyjne przeszukiwanie folderów
    for class_path in input_dir.iterdir():
        if class_path.is_dir():
            # Tworzymy folder dla klasy w katalogu wyjściowym
            output_class_path = output_dir / class_path.name
            output_class_path.mkdir(exist_ok=True)

            # Pobierz wszystkie obrazy PNG z folderu klasy
            png_files = list(class_path.glob('*.png'))

            for img_path in png_files:
                try:
                    # Wczytaj obraz
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"Nie można odczytać obrazu: {img_path}")
                        continue

                    # Zapisz oryginalny obraz
                    orig_output_path = output_class_path / img_path.name
                    cv2.imwrite(str(orig_output_path), img)

                    # Wygeneruj i zapisz augmentowane wersje
                    augmented_images = augment_image(img)
                    for i, aug_img in enumerate(augmented_images):
                        aug_path = output_class_path / f"{img_path.stem}_aug{i + 1}.png"
                        cv2.imwrite(str(aug_path), aug_img)

                    print(f"Augmentowano: {img_path}")
                except Exception as e:
                    print(f"Błąd podczas przetwarzania {img_path}: {e}")

    return output_dir


def split_train_test(input_dir, train_dir, test_dir, test_ratio=0.2):
    input_dir = Path(input_dir)
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    train_dir.mkdir(exist_ok=True, parents=True)
    test_dir.mkdir(exist_ok=True, parents=True)

    # Przeszukaj każdy folder klasy
    for class_path in input_dir.iterdir():
        if class_path.is_dir():
            # Utwórz odpowiednie foldery w train i test
            train_class_dir = train_dir / class_path.name
            test_class_dir = test_dir / class_path.name

            train_class_dir.mkdir(exist_ok=True)
            test_class_dir.mkdir(exist_ok=True)

            # Pobierz wszystkie obrazy z folderu klasy
            image_files = list(class_path.glob('*.png'))
            random.shuffle(image_files)

            # Wyznacz punkt podziału
            split_idx = int(len(image_files) * (1 - test_ratio))
            train_images = image_files[:split_idx]
            test_images = image_files[split_idx:]

            # Kopiuj obrazy do odpowiednich folderów
            for img_path in train_images:
                shutil.copy2(img_path, train_class_dir / img_path.name)

            for img_path in test_images:
                shutil.copy2(img_path, test_class_dir / img_path.name)

            print(
                f"Klasa {class_path.name}: {len(train_images)} obrazów treningowych, {len(test_images)} obrazów testowych")

    return train_dir, test_dir


def create_yolo_labels(dataset_dir, class_dict=None):
    dataset_dir = Path(dataset_dir)

    # Jeśli nie dostarczono słownika klas, utwórz go
    if class_dict is None:
        class_dirs = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
        class_dict = {class_name: idx for idx, class_name in enumerate(sorted(class_dirs))}

    # Dla każdego folderu klasy
    for class_name, class_id in class_dict.items():
        class_path = dataset_dir / class_name
        if not class_path.is_dir():
            continue

        image_files = list(class_path.glob("*.png"))

        for img_path in image_files:
            # Pobierz wymiary obrazu
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size

                # Oblicz znormalizowane współrzędne bounding boxa w formacie YOLO:
                # <class_id> <center_x> <center_y> <width> <height>
                center_x = 0.5  # Środek obrazu w osi X
                center_y = 0.5  # Środek obrazu w osi Y
                box_width = 0.8  # 80% szerokości obrazu (po odjęciu 20%)
                box_height = 0.8  # 80% wysokości obrazu (po odjęciu 20%)

                # Przygotuj zawartość pliku tekstowego
                label_content = f"{class_id} {center_x} {center_y} {box_width} {box_height}"

                # Zapisz plik etykiety (zamień rozszerzenie .png na .txt)
                label_path = img_path.with_suffix(".txt")
                with open(label_path, "w") as f:
                    f.write(label_content)
            except Exception as e:
                print(f"Błąd podczas przetwarzania {img_path}: {e}")

    return class_dict


def create_yaml_file(train_dir, test_dir, class_dict, output_path):
    output_path = Path(output_path)

    yaml_content = f"""# YOLOv8 Dataset Configuration
path: {output_path.parent}  # Ścieżka do głównego folderu
train: {train_dir.name}  # Ścieżka do folderu treningowego
val: {test_dir.name}  # Ścieżka do folderu testowego

# Classes
nc: {len(class_dict)}  # Liczba klas
names: {list(class_dict.keys())}  # Nazwy klas
"""

    with open(output_path, "w") as f:
        f.write(yaml_content)

    print(f"Utworzono plik konfiguracyjny: {output_path}")
    return output_path


def main():
    # Ścieżki
    base_dir = Path("../dataset")
    source_dir = base_dir / "ELE_DATASET"
    prepared_dir = base_dir / "ele_dataset_prepaerd"
    augmented_dir = base_dir / "ele_dataset_augmented"
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    yaml_path = base_dir / "data.yaml"

    # 1. Reorganizacja obrazów z oryginalnego datasetu
    print("Krok 1: Reorganizacja obrazów...")
    reorganize_images(source_dir, prepared_dir)

    # 2. Augmentacja obrazów
    print("Krok 2: Augmentacja obrazów...")
    augment_images(prepared_dir, augmented_dir, augmentations_per_image=5)

    # 3. Podział na zbiory treningowy i testowy
    print("Krok 3: Podział na zbiory treningowy i testowy...")
    train_dir, test_dir = split_train_test(augmented_dir, train_dir, test_dir, test_ratio=0.2)

    # 4. Utwórz słownik klas
    class_dirs = [d.name for d in augmented_dir.iterdir() if d.is_dir()]
    class_dict = {class_name: idx for idx, class_name in enumerate(sorted(class_dirs))}

    # 5. Generowanie etykiet YOLO dla zbioru treningowego i testowego
    print("Krok 4: Generowanie etykiet YOLO...")
    create_yolo_labels(train_dir, class_dict)
    create_yolo_labels(test_dir, class_dict)

    # 6. Utworzenie pliku konfiguracyjnego YAML
    print("Krok 5: Tworzenie pliku konfiguracyjnego YAML...")
    create_yaml_file(train_dir, test_dir, class_dict, yaml_path)

    print("\nProcesowanie zakończone.")
    print(f"Zbiór treningowy: {train_dir}")
    print(f"Zbiór testowy: {test_dir}")
    print(f"Plik konfiguracyjny: {yaml_path}")


if __name__ == "__main__":
    main()