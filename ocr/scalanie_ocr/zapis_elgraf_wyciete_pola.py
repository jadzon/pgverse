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
import onnxruntime as ort
import json
import time
import numpy as np
# Funkcja obejścia problemu importu flash_attn w modelach Hugging
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = original_get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports
# (opcjonalnie) ustawienia sesji ORT
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
sess_opts.inter_op_num_threads = 1
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.log_severity_level = 1
# inicjalizacja z GPU i szybkim procesorem obrazu
p2t = Pix2Text.from_config(
    provider="CUDAExecutionProvider",
    session_options=sess_opts,
    use_fast=True
)
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
def detect_with_pix2text(image: Image.Image) -> list:
    return p2t.recognize(image, file_type="text_formula", return_text=False)

def get_bounding_box(item):
    """
    Zwraca [x1, y1, x2, y2] na podstawie pola 'position' w zwróconym obiekcie Pix2Text,
    albo None jeśli nie ma pozycji.
    """
    if 'position' in item:
        pts = item['position']            # lista punktów [[x,y],…]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None
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
    • Wszystkie boksy z Hugging Face przepuszczamy bez zmian.
    • Boks z Detectrona dodajemy tylko wtedy, gdy
      (pole przecięcia / pole boksa Detectrona) ≤ 0.5.
    """

    def overlap_ratio(big, small):
        """zwraca (intersect_area / area_small)"""
        x1 = max(big[0], small[0]); y1 = max(big[1], small[1])
        x2 = min(big[2], small[2]); y2 = min(big[3], small[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_small = (small[2] - small[0]) * (small[3] - small[1])
        return inter / area_small if area_small else 0.0

    KEEP_THR = 0.6         # 70 %

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
    # 1. Przygotowanie obrazu
    image = ImageOps.autocontrast(image)

    # 2. Detekcje
    hugging_dets   = detect_with_hugging(image)
    detectron_dets = detect_with_detectron(image)
    pix2text_dets  = detect_with_pix2text(image)
    merged         = merge_detections(hugging_dets, detectron_dets)

    # 3. Usuń duże figury zawierające >= 2 mniejszych figure
    def contains(inner, outer):
        return (inner[0] >= outer[0] and inner[1] >= outer[1]
                and inner[2] <= outer[2] and inner[3] <= outer[3])

    clean_merged = []
    for bbox, label in merged:
        if label == "figure":
            count_inside = sum(
                1 for ob, lbl in merged
                if lbl == "figure" and ob != bbox and contains(ob, bbox)
            )
            if count_inside >= 2:
                continue
        clean_merged.append((bbox, label))
    merged = clean_merged

    # 4. Przygotuj katalogi wyników
    wzory_dir   = os.path.join(results_dir, "wzory")
    tables_dir  = os.path.join(results_dir, "tabele")
    figures_dir = os.path.join(results_dir, "figury")
    os.makedirs(wzory_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 5. Wyodrębnij bounding boxy formuł
    formula_boxes = []
    for it in pix2text_dets:
        if it.get("type") != "isolated":
            continue
        bb = get_bounding_box(it)
        if bb is None:
            continue
        formula_boxes.append((tuple(map(int, bb)), "formula"))

    # 6. Dalsze filtrowanie boxów wg udziału powierzchni formuł
    def area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    def intersect(box1, box2):
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        return (x1, y1, x2, y2) if x2> x1 and y2> y1 else None

    filtered = []
    for bbox, label in merged:
        # odrzuć, jeśli w table/figure >=2 formuły
        if label in ["figure", "table"]:
            overlaps = sum(
                1 for f_box, _ in formula_boxes
                if intersect(bbox, f_box)
            )
            if overlaps >= 2:
                continue
        # zachowaj figury nakładające się na 1 formułę
        if label == "figure" and any(intersect(bbox, f_box) for f_box, _ in formula_boxes):
            filtered.append((bbox, label))
            continue
        # odrzuć, jeśli formuły zajmują >20% powierzchni
        if label in ["figure", "table"]:
            tot = area(bbox)
            frm = sum(area(intersect(bbox, f_box) or (0,0,0,0)) for f_box, _ in formula_boxes)
            if tot>0 and frm/tot>0.2:
                continue
        filtered.append((bbox, label))

    # 7. Dystrybucja finalnych detekcji
    final_detections = filtered + formula_boxes

    # 8. Wklej białe tło pod każdy wykryty obszar
    for bbox, _ in final_detections:
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        white = Image.new("RGB", (w, h), (255, 255, 255))
        image.paste(white, (x1, y1))

    # 9. Rysuj obramowania i etykiety
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    count_table = count_fig = count_formula = 0

    for bbox, label in final_detections:
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1

        if label != "formula":
            rect = patches.Rectangle((x1, y1), w, h,
                                     linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1, label, color="red", fontsize=12,
                    bbox=dict(facecolor="yellow", alpha=0.5))

        # 10. Zapis wycinków
        if label == "table":
            count_table += 1
            fn = f"{output_prefix}_page{page_idx}_table{count_table}.png"
            image.crop((x1, y1, x2, y2)).save(os.path.join(tables_dir, fn))
        elif label == "figure":
            count_fig += 1
            fn = f"{output_prefix}_page{page_idx}_figure{count_fig}.png"
            image.crop((x1, y1, x2, y2)).save(os.path.join(figures_dir, fn))
        elif label == "formula":
            count_formula += 1
            fn = f"{output_prefix}_page{page_idx}_formula_{count_formula}.png"
            image.crop((x1, y1, x2, y2)).save(os.path.join(wzory_dir, fn))
        print(f"Zapisano {label}: {fn}")

    # 11. Zapis końcowego obrazu ze „dziurami”
    result_name = f"{output_prefix}_page{page_idx}_result.png"
    result_path = os.path.join(results_dir, result_name)
    plt.title(f"Wynik dla {output_prefix} (strona {page_idx})")
    plt.savefig(result_path)
    plt.close(fig)
    print(f"Zapisano wynikowy obraz: {result_name}")



""""def main():
    root = tk.Tk()
    root.withdraw()

    filetypes = [
        ("PDF files", "*.pdf"),
        ("Image files", ("*.png","*.jpg","*.jpeg"))
    ]
    selected_files = filedialog.askopenfilenames(
        title="Wybierz pliki PDF lub obrazy",
        initialdir=os.getcwd(),
        filetypes=filetypes
    )
    if not selected_files:
        print("Nie wybrano plików. Zakończono.")
        return

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor

    for selected_file in selected_files:
        basename = os.path.basename(selected_file)
        output_prefix = os.path.splitext(basename)[0]
        print(f"\n===== Przetwarzam: {basename} =====")

        _, ext = os.path.splitext(selected_file)
        ext = ext.lower()

        if ext == ".pdf":
            pages = convert_from_path(selected_file, dpi=300)
            args = [
                (page, output_prefix, results_dir, idx)
                for idx, page in enumerate(pages, start=1)
            ]
            with ThreadPoolExecutor(max_workers=4) as exe:
                exe.map(lambda tpl: (print(f"Strona {tpl[3]}…"), process_image(*tpl)), args)

        elif ext in [".png", ".jpg", ".jpeg"]:
            image = Image.open(selected_file).convert("RGB")
            process_image(image, output_prefix, results_dir)

        else:
            print(f"{basename}: format nieobsługiwany.")"""
def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 1. Przygotuj listę k1.pdf...k10.pdf
    selected_files = []
    for i in range(1, 11):
        fn = f"k{i}.pdf"
        if os.path.isfile(fn):
            selected_files.append(fn)
        else:
            print(f"Uwaga: {fn} nie istnieje, pomijam.")
    if not selected_files:
        print("Brak plików k1–k10. Kończę.")
        return

    # 2. Utworzenie katalogu wyników
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 3. Dla każdego pliku PDF: konwersja stron i przetwarzanie w wątkach
    for selected_file in selected_files:
        basename = os.path.basename(selected_file)
        output_prefix = os.path.splitext(basename)[0]
        print(f"\n===== Przetwarzam: {basename} =====")

        # Konwertuj PDF na listę obrazów
        pages = convert_from_path(selected_file, dpi=300)

        # Równoległe przetwarzanie każdej strony
        with ThreadPoolExecutor(max_workers=2) as exe:
            # Budujemy dict: future -> page_idx
            futures = {
                exe.submit(process_image, page, output_prefix, results_dir, idx): idx
                for idx, page in enumerate(pages, start=1)
            }

            # Odbieramy wyniki w miarę zakończenia tasków
            for fut in as_completed(futures):
                page_idx = futures[fut]
                try:
                    fut.result()
                    print(f"  Strona {page_idx} przetworzona")
                except Exception as e:
                    print(f" Błąd przy przetwarzaniu strony {page_idx}:", e)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Czas wykonania programu: {time.time() - start_time:.2f} sekundy")

