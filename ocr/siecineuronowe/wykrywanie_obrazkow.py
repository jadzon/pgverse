import cv2
import numpy as np
from pdf2image import convert_from_path
import layoutparser as lp
import csv

# 1. Konwertujemy PDF na listę obrazów PIL
pages = convert_from_path("pierdola1.pdf", dpi=150)

# 2. Inicjalizujemy model layoutu (PubLayNet)
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    model_path="model_final.pth",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    device="cuda"  # "cpu" jeżeli bez CUDA
)


for idx, page_pil in enumerate(pages):
        page_cv = cv2.cvtColor(np.array(page_pil), cv2.COLOR_RGB2BGR)

        # 4. Wykrycie layoutu na stronie
        layout = model.detect(page_cv)


        text_blocks = [l for l in layout if l.type == "Text"]
        figure_blocks = [l for l in layout if l.type == "Figure"]

        # 6. Rysujemy bounding boxy
        # 6a. Tekst (niebieski)
        for block in text_blocks:
            x1, y1, x2, y2 = map(int, block.block.coordinates)
            cv2.rectangle(page_cv, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)


        # 6b. Obrazy (czerwony)
        for block in figure_blocks:
            x1, y1, x2, y2 = map(int, block.block.coordinates)
            cropped_figure = page_cv[y1:y2, x1:x2]

            # Zapisz wycinek
            figure_path = f"figure_{idx}.png"
            cv2.imwrite(figure_path, cropped_figure)
            print(f"Wycięto figurę {idx} -> {figure_path}")

            # (Opcjonalnie) Rysujemy bounding box na pełnej stronie
            cv2.rectangle(page_cv, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        # 7. Zapisujemy wynik do pliku
        output_path = f"output_layout_page_{idx}.jpg"
        cv2.imwrite(output_path, page_cv)
        print(f"Strona {idx} → zapisano: {output_path}")
