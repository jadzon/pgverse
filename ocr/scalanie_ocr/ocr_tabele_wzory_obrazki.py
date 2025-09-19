import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps
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
from pix2text import Pix2Text
from PIL import Image
import json
import numpy as np
# Funkcja obejścia problemu importu flash_attn w modelach Hugging
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """
    Funkcjonalność:
        Poprawia listę importów w Transformers, usuwając wpis "flash_attn",
        aby uniknąć błędów przy ładowaniu modeli.

    Args:
        filename - ścieżka do pliku źródłowego (str lub Path)

    Returns:
        list[str] - lista importów bez "flash_attn"
    """
    imports = original_get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports
p2t = Pix2Text.from_config()
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
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    device="cuda"  # lub "cpu", jeśli GPU nie jest dostępne
)
def get_bounding_box(item):
    """
    Funkcjonalność:
        Wyznacza prostokąt [x1, y1, x2, y2] na podstawie pola 'position'
        w obiekcie zwróconym przez Pix2Text.

    Args:
        item - obiekt ze słownikiem zawierającym klucz 'position'

    Returns:
        list[int] | None - współrzędne bounding boxa lub None
    """
    if 'position' in item:
        pts = item['position']            # lista punktów [[x,y],…]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None
def detect_with_hugging(image: Image.Image) -> list:
    """
    Funkcjonalność:
        Wykrywa tabele i figury na obrazie przy pomocy modelu Hugging Face.

    Args:
        image - obraz wejściowy (PIL.Image)

    Returns:
        list - lista wykryć w formacie (bbox, label),
               gdzie label to "table" lub "figure"
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
    Funkcjonalność:
        Wykrywa figury na obrazie przy użyciu Detectron2 (LayoutParser).

    Args:
        image - obraz wejściowy (PIL.Image)

    Returns:
        list - lista wykryć w formacie (bbox, "figure")
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
    Funkcjonalność:
        Łączy wykrycia z Hugging i Detectron:
        - wszystkie boksy z Hugging przepuszczane bez zmian,
        - boks z Detectrona dodawany tylko wtedy, gdy
          pokrycie z boksem Hugging ≤ 70%.

    Args:
        hugging_dets - lista wykryć (bbox, label) z Hugging
        detectron_dets - lista wykryć (bbox, label) z Detectron2

    Returns:
        list - lista połączonych wykryć (bbox, label)
    """

    def overlap_ratio(big, small):
        """
        Funkcjonalność:
            Oblicza stosunek pola części wspólnej dwóch prostokątnych bboxów
            do pola mniejszego z nich.

        Args:
            big - współrzędne bboxa [x1, y1, x2, y2]
            small - współrzędne bboxa [x1, y1, x2, y2]

        Returns:
            float - proporcja pokrycia (0–1)
        """
        x1 = max(big[0], small[0]); y1 = max(big[1], small[1])
        x2 = min(big[2], small[2]); y2 = min(big[3], small[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_small = (small[2] - small[0]) * (small[3] - small[1])
        return inter / area_small if area_small else 0.0

    KEEP_THR = 0.7          # 70 %

    merged = list(hugging_dets)
    for d_bbox, d_label in detectron_dets:
        # jeśli z dowolnym boksem HF zachodzi > 70 %, pomijamy
        if any(overlap_ratio(d_bbox, h_bbox) > KEEP_THR for h_bbox, _ in hugging_dets):
            continue
        merged.append((d_bbox, d_label))

    return merged
def process_image(image: Image.Image,
                  output_prefix: str,
                  results_dir: str,
                  page_idx: int = None):
    """
    Funkcjonalność:
        Przetwarza obraz w celu detekcji i ekstrakcji:
        1. Autokontrast obrazu.
        2. Detekcja tabel i figur (Hugging + Detectron).
        3. Rozpoznawanie wzorów (Pix2Text).
        4. Rysowanie bounding boxów i zapisywanie wycinków.
        5. Zapis finalnego obrazu z naniesionymi wynikami.

    Args:
        image - obraz wejściowy (PIL.Image)
        output_prefix - prefiks nazw plików wynikowych
        results_dir - katalog na wyniki
        page_idx - (opcjonalne) numer strony w PDF

    Returns:
        None
    """
    # 1) Autokontrast
    image = ImageOps.autocontrast(image)

    # 2) Detekcja tabel i figur
    hugging_dets   = detect_with_hugging(image)
    detectron_dets = detect_with_detectron(image)
    merged         = merge_detections(hugging_dets, detectron_dets)

    # 3) Przygotuj katalogi
    wzory_dir   = os.path.join(results_dir, "wzory")
    tables_dir  = os.path.join(results_dir, "tabele")
    figures_dir = os.path.join(results_dir, "figury")
    os.makedirs(wzory_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 4) Stwórz jedno płótno
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)

    # 5) Rozpoznanie wzorów i rysowanie niebieskich BB tylko dla izolowanych
    items = p2t.recognize(image, file_type="text_formula", return_text=False)
    for idx, it in enumerate(items):
        # rysujemy tylko izolowane wzory
        if it.get("type") != "isolated":
            continue
        bb = get_bounding_box(it)
        if bb is None:
            continue
        x1, y1, x2, y2 = map(int, bb)

        # narysuj niebieski bounding box dla izolowanego wzoru
        rect_f = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor="blue", facecolor="none")
        ax.add_patch(rect_f)
        ax.text(x1, y2, "formula", color="blue", fontsize=12,
                bbox=dict(facecolor="yellow", alpha=0.5))

        # zapisz wycinek izolowanego wzoru
        fn = (f"{output_prefix}_page{page_idx}_formula_{idx}.png"
              if page_idx is not None
              else f"{output_prefix}_formula_{idx}.png")
        image.crop((x1, y1, x2, y2)).save(os.path.join(wzory_dir, fn))
        print(f"Zapisano wzór: {fn}")

    # 6) Rysowanie czerwonych BB tabel i figur oraz zapis wycinków
    count_table, count_fig = 0, 0
    for bbox, label in merged:
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1

        # 6a) narysuj czerwony bounding box
        rect = patches.Rectangle((x1, y1), w, h,
                                 linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1, label, color="red", fontsize=12,
                bbox=dict(facecolor="yellow", alpha=0.5))

        # 6b) zapisz wycinek
        if label == "table":
            count_table += 1
            fn = (f"{output_prefix}_page{page_idx}_table{count_table}.png"
                  if page_idx is not None
                  else f"{output_prefix}_table{count_table}.png")
            image.crop((x1, y1, x2, y2)).save(os.path.join(tables_dir, fn))
            print(f"Zapisano tabelę: {fn}")

        elif label == "figure":
            # pomiń figurę zajmującą całą stronę
            if w > 0.94 * image.width and h > 0.94 * image.height:
                continue
            count_fig += 1
            fn = (f"{output_prefix}_page{page_idx}_figure{count_fig}.png"
                  if page_idx is not None
                  else f"{output_prefix}_figure{count_fig}.png")
            image.crop((x1, y1, x2, y2)).save(os.path.join(figures_dir, fn))
            print(f"Zapisano figurę: {fn}")

    # 7) Zapis finalnego obrazka
    result_name = (f"{output_prefix}_page{page_idx}_result.png"
                   if page_idx is not None
                   else f"{output_prefix}_result.png")
    result_path = os.path.join(results_dir, result_name)
    plt.title(f"Wynik dla {output_prefix}" +
              (f" (strona {page_idx})" if page_idx is not None else ""))
    plt.savefig(result_path)
    plt.close(fig)
    print(f"Zapisano wynikowy obraz: {result_name}")


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
        pages = convert_from_path(selected_file, dpi=400)
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
