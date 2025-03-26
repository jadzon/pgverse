import os
import cv2
from pathlib import Path

#PROGRAM UŻYTY DO ODWRÓCENIA KOLORÓW I ZMIANY ROZSZERZENIA W DATASECIE Z  
#https://www.kaggle.com/datasets/moodrammer/handdrawn-circuit-schematic-components/data?select=SolvaDataset_200_v3
def invert_image(image_path, output_path):

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Nie udało się wczytać obrazu: {image_path}")
        return False

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    inverted = cv2.bitwise_not(gray)

    _, binary = cv2.threshold(inverted, 128, 255, cv2.THRESH_BINARY)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = cv2.imwrite(str(output_path), binary)

    if result:
        print(f"Zapisano odwrócony obraz: {output_path}")
        return True
    else:
        print(f"Nie udało się zapisać obrazu: {output_path}")
        return False


def process_dataset(input_dir, output_dir):

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(exist_ok=True)

    total_images = 0
    processed_images = 0

    for path in input_path.glob('**/*'):
        if path.is_dir():
            continue

        if path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            continue

        total_images += 1

        relative_path = path.relative_to(input_path)

        output_file_path = output_path / (relative_path.with_suffix('.png'))

        if invert_image(path, output_file_path):
            processed_images += 1

    print(f"\nZakończono przetwarzanie zbioru danych.")
    print(f"Przetworzono {processed_images} z {total_images} obrazów.")


if __name__ == "__main__":

    input_dataset = "dataset"
    output_dataset = "dataset_new"

    process_dataset(input_dataset, output_dataset)