import os
import tkinter as tk
from tkinter import filedialog
import requests
from PIL import Image
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports as original_get_imports
from unittest.mock import patch


# Funkcja obejścia problemu importu flash_attn
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = original_get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained("yifeihu/TF-ID-base", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("yifeihu/TF-ID-base", trust_remote_code=True)

# Ustawiamy prompt, który informuje model, że chcemy detekcji obiektów (tabel)
prompt = "<OD>"


def process_image(image, output_prefix, results_dir, page_idx=None):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text,
        task="<OD>",
        image_size=(image.width, image.height)
    )

    tables_dir = os.path.join(results_dir, "tabele")
    os.makedirs(tables_dir, exist_ok=True)

    if "<OD>" in result and "bboxes" in result["<OD>"] and "labels" in result["<OD>"]:
        table_count = 0
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)
        for bbox, label in zip(result["<OD>"]["bboxes"], result["<OD>"]["labels"]):

            if label.lower() == "table":
                x1, y1, x2, y2 = bbox  # bbox w formacie [x1, y1, x2, y2]
                width = x2 - x1
                height = y2 - y1
                # Pomijamy detekcje, które zajmują prawie całą stronę
                if width > 0.9 * image.width and height > 0.9 * image.height:
                    print("Pomijam detekcję, bo bounding box jest zbyt duży (cała strona).")
                    continue

                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, label, fontsize=12, color='red', bbox=dict(facecolor='yellow', alpha=0.5))
                # Wycinamy tabelę
                cropped_table = image.crop((x1, y1, x2, y2))
                table_count += 1
                if page_idx is not None:
                    crop_path = os.path.join(tables_dir, f"{output_prefix}_page{page_idx}_table{table_count}.png")
                else:
                    crop_path = os.path.join(tables_dir, f"{output_prefix}_table{table_count}.png")
                cropped_table.save(crop_path)
                print(f"Zapisano tabelę: {crop_path}")
        plt.title(f"Wynik dla {output_prefix}" + (f" (strona {page_idx})" if page_idx is not None else ""))
        plt.show()
    else:
        print("Wynik nie zawiera oczekiwanej struktury .")


def main():
    root = tk.Tk()
    root.withdraw()

    filetypes = [
        ("PDF files", "*.pdf"),
        ("Image files", ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"))
    ]
    selected_file = filedialog.askopenfilename(
        title="Wybierz plik PDF lub obraz (JPG/PNG)",
        initialdir=os.getcwd(),
        filetypes=filetypes
    )
    if not selected_file:
        print("Nie wybrano pliku. Zakończono.")
        return

    _, ext = os.path.splitext(selected_file)
    ext = ext.lower()

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    output_prefix = os.path.splitext(os.path.basename(selected_file))[0]

    if ext == ".pdf":
        pages = convert_from_path(selected_file, dpi=300)
        for idx, page in enumerate(pages, start=1):
            print(f"Przetwarzam stronę {idx}...")
            process_image(page, output_prefix, results_dir, page_idx=idx)
    elif ext in [".png", ".jpg", ".jpeg"]:
        image = Image.open(selected_file).convert("RGB")
        process_image(image, output_prefix, results_dir)
    else:
        print("Nieobsługiwany format pliku.")


if __name__ == "__main__":
    main()
