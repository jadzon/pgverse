import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Backend bez GUI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports as original_get_imports
from unittest.mock import patch
import layoutparser as lp
import cv2
import numpy as np
from pdf2image import convert_from_path

# Funkcja obejścia problemu importu flash_attn w modelach Hugging
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = original_get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# Ładujemy modele Hugging Face z pominięciem flash_attn
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    hugging_model = AutoModelForCausalLM.from_pretrained("yifeihu/TF-ID-base", trust_remote_code=True)
    hugging_processor = AutoProcessor.from_pretrained("yifeihu/TF-ID-base", trust_remote_code=True)

# Ustawienie promptu – model Hugging wykrywa obiekty (tabele i figury)
prompt = "<OD>"

# Ładowanie modelu Detectron2 przez LayoutParser (do wykrywania figur)
detectron_model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    model_path="model_final.pth",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
    device="cuda"  # lub "cpu", jeśli GPU nie jest dostępne
)

def detect_with_hugging(image: Image.Image) -> list:
    """
    Detekcja obiektów przy użyciu modelu Hugging.
    Wykrywa zarówno tabele, jak i figury.
    """
    inputs = hugging_processor(text=prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = hugging_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
    generated_text = hugging_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = hugging_processor.post_process_generation(
        generated_text,
        task="<OD>",
        image_size=(image.width, image.height)
    )
    detections = []
    if "<OD>" in result and "bboxes" in result["<OD>"] and "labels" in result["<OD>"]:
        for bbox, label in zip(result["<OD>"]["bboxes"], result["<OD>"]["labels"]):
            if label.lower() in ["table", "figure"]:
                detections.append((bbox, label.lower()))
    return detections

def detect_with_detectron(image: Image.Image) -> list:
    """
    Wykrywanie figur przy użyciu Detectrona (LayoutParser).

    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    layout = detectron_model.detect(image_cv)
    detections = []
    for block in layout:
        if block.type == "Figure":
            x1, y1, x2, y2 = map(int, block.block.coordinates)
            detections.append(((x1, y1, x2, y2), "figure"))
    return detections

def merge_detections(hugging_dets: list, detectron_dets: list) -> list:
    """
    Łączy wyniki:
      - Wykrycia tabel (label "table") pochodzą wyłącznie z Hugging.
      - Wykrycia figur łączymy z obu źródeł, przy czym jeśli dowolny bbox z Hugging
        (niezależnie czy table czy figure) pokrywa centralny punkt wykrycia z Detectrona,
        to wykrycie z Detectrona jest pomijane.
    """
    final_detections = []
    for bbox, label in hugging_dets:
        final_detections.append((bbox, label))
    for d_bbox, d_label in detectron_dets:
        add_det = True
        x1_d, y1_d, x2_d, y2_d = d_bbox
        center_d = ((x1_d + x2_d) / 2, (y1_d + y2_d) / 2)
        for h_bbox, _ in hugging_dets:
            x1_h, y1_h, x2_h, y2_h = h_bbox
            if x1_h <= center_d[0] <= x2_h and y1_h <= center_d[1] <= y2_h:
                add_det = False
                break
        if add_det:
            final_detections.append((d_bbox, d_label))
    return final_detections

def process_image(image: Image.Image, output_prefix: str, results_dir: str, page_idx: int = None):
    """
    Przetwarzanie obrazu:
      1. Detekcja przy użyciu modelu Hugging (tabele i figury).
      2. Detekcja figur przy użyciu Detectrona.
      3. Łączenie wyników – tabele mają priorytet z Hugging; figury uzupełniane z Detectrona.
      4. Zapis wykrytych fragmentów do osobnych folderów: tabele i figury.
      5. Jeżeli figura zajmuje prawie całą stronę, pomijamy ją.
    """
    hugging_detections = detect_with_hugging(image)
    print("Detekcje z Hugging:", hugging_detections)

    detectron_detections = detect_with_detectron(image)
    print("Detekcje z Detectrona:", detectron_detections)

    merged = merge_detections(hugging_detections, detectron_detections)
    print("Połączone wykrycia:", merged)

    # Utworzenie folderów na wycięte obszary
    tables_dir = os.path.join(results_dir, "tabele")
    figures_dir = os.path.join(results_dir, "figury")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)


    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    count_table = 0
    count_figure = 0

    for bbox, label in merged:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Rysowanie bounding boxa i etykiety na obrazie
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, label, fontsize=12, color='red', bbox=dict(facecolor='yellow', alpha=0.5))

        if label == "table":
            count_table += 1
            if page_idx is not None:
                crop_path = os.path.join(tables_dir, f"{output_prefix}_page{page_idx}_table{count_table}.png")
            else:
                crop_path = os.path.join(tables_dir, f"{output_prefix}_table{count_table}.png")
            cropped = image.crop((x1, y1, x2, y2))
            cropped.save(crop_path)
            print(f"Zapisano tabelę: {crop_path}")
        elif label == "figure":
            # Sprawdzenie, czy figura nie zajmuje prawie całej strony (90% szerokości i 90% wysokości)
            if width > 0.9 * image.width and height > 0.9 * image.height:
                print("Pomijam wykrycie figury, bo zajmuje prawie całą stronę.")
                continue
            count_figure += 1
            if page_idx is not None:
                crop_path = os.path.join(figures_dir, f"{output_prefix}_page{page_idx}_figure{count_figure}.png")
            else:
                crop_path = os.path.join(figures_dir, f"{output_prefix}_figure{count_figure}.png")
            cropped = image.crop((x1, y1, x2, y2))
            cropped.save(crop_path)
            print(f"Zapisano figurę: {crop_path}")

    # Zapis wynikowego obrazu z naniesionymi wykryciami
    if page_idx is not None:
        plot_path = os.path.join(results_dir, f"{output_prefix}_page{page_idx}_result.png")
    else:
        plot_path = os.path.join(results_dir, f"{output_prefix}_result.png")
    plt.title(f"Wynik dla {output_prefix}" + (f" (strona {page_idx})" if page_idx is not None else ""))
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Zapisano wynikowy obraz: {plot_path}")

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
