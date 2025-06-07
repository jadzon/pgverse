#!/usr/bin/env python3
"""
Przetwarza wszystkie obrazy tabel w strukturze:
  ksiazki/<book>/detekcje/tabele/*.png
i zapisuje CSV do:
  ksiazki/<book>/rezultaty/tabele/*.csv

Dodatkowo – przed samym OCR powiększamy każdą tabelę (np. 2× lub 3×),
aby uzyskać lepszą jakość rozpoznawania tekstu.
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR
from PIL import Image as PILImage  # do ewentualnego zapisu tymczasowego
import tempfile

# Główne katalogi i próg
BOOKS_DIR = Path("ksiazki")
SCORE_THRESHOLD = 0.9

# Inicjalizacja OCR
ocr_paddle = PaddleOCR(lang="pl", use_angle_cls=True, use_gpu=False)
ocr_tess = TesseractOCR(lang="pol")


def paddle_score_above_threshold(result, threshold=SCORE_THRESHOLD) -> float:
    good = sum(1 for box in result if box[1][1] >= threshold)
    total = len(result)
    print(f" {good}/{total} boxów ≥ {threshold}")
    return (good / total) if total else 0.0


def preprocess_for_paddle_upscaled(path: Path, scale: int = 2) -> np.ndarray:
    """
    Wczytuje obraz z path jako grayscale, powiększa go (scale x),
    a następnie aplikuje blur + threshold i zwraca RGB ready for PaddleOCR.
    """
    # 1) Wczytaj w skali szarości
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nie udało się wczytać obrazu: {path}")
    # 2) Powiększ obraz (INTER_CUBIC dla lepszego efektu)
    h, w = img.shape
    img_up = cv2.resize(img, (w * scale, h * scale),
                        interpolation=cv2.INTER_CUBIC)
    # 3) Blur + threshold (jak wcześniej)
    blur = cv2.GaussianBlur(img_up, (3, 3), 0)
    _, bin_img = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    # 4) Konwersja do RGB
    return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)


def paddle_grid_group(result, y_thresh=20, x_thresh=30) -> list:
    # Jeśli nie ma wykrytych boxów, zwracamy pusty grid
    if not result:
        return []

    cells = []
    for box, (text, _) in result:
        x_c = sum(p[0] for p in box) / 4
        y_c = sum(p[1] for p in box) / 4
        cells.append({'x': x_c, 'y': y_c, 'text': text})
    cells.sort(key=lambda c: (c['y'], c['x']))

    rows, current, last_y = [], [], None
    for cell in cells:
        if last_y is None or abs(cell['y'] - last_y) <= y_thresh:
            current.append(cell)
        else:
            rows.append(current)
            current = [cell]
        last_y = cell['y']
    if current:
        rows.append(current)

    # teraz rows nie jest puste, możemy liczyć max_cols
    max_cols = max(len(r) for r in rows)

    # budujemy finalny grid z dopełnieniem pustymi komórkami
    grid = []
    for row in rows:
        texts = [c['text'] for c in row]
        grid.append(texts + [""] * (max_cols - len(texts)))
    return grid


if __name__ == "__main__":
    # Dla każdej książki
    for book_dir in sorted(BOOKS_DIR.iterdir()):
        if not book_dir.is_dir():
            continue
        input_dir = book_dir / "detekcje" / "tabele"
        output_dir = book_dir / "rezultaty" / "tabele"
        if not input_dir.exists():
            print(f"Brak katalogu: {input_dir}, pomijam {book_dir.name}.")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        # Przetwarzanie każdego pliku obrazka
        for img_path in sorted(input_dir.glob("*.*")):
            fname = img_path.stem
            print(f"\nPrzetwarzanie tabel: {book_dir.name}/{fname}")

            # 1) Powiększenie obrazu tabeli
            try:
                SCALE = 2
                upscaled_np = preprocess_for_paddle_upscaled(img_path, scale=SCALE)
            except Exception as e:
                print(f"  Błąd podczas powiększania {img_path.name}: {e}")
                continue

            # 2) OCR Paddle na powiększonym obrazie
            raw = ocr_paddle.ocr(upscaled_np, cls=False)
            if not raw or raw[0] is None:
                print("  Brak wyników OCR Paddle → fallback")
                grid = paddle_grid_group([])
                out_csv = output_dir / f"{fname}_paddle.csv"
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerows(grid)
                print(f"   Zapisano: {out_csv}")
                continue

            # Jeśli jest coś w raw, to bierzemy pierwszy element
            result_paddle = raw[0]

            # 3) Obliczamy score
            score = paddle_score_above_threshold(result_paddle)

            if score >= SCORE_THRESHOLD:
                print(f"  img2table (score={score:.2f})")
                with tempfile.NamedTemporaryFile(suffix=f"_{fname}_upscaled.png", delete=False) as tmpf:
                    tmp_path = Path(tmpf.name)
                    up_img_pil = PILImage.fromarray(upscaled_np)
                    up_img_pil.save(str(tmp_path))
                    up_img_pil.close()

                doc = Img2TableImage(str(tmp_path), detect_rotation=True)
                tables = doc.extract_tables(ocr=ocr_tess)
                try:
                    os.remove(str(tmp_path))
                except OSError:
                    pass

                if not tables:
                    print("  Brak tabel (pomimo dobrego score).")
                    continue

                for idx, table in enumerate(tables, start=1):
                    df = table.df.copy()
                    ocr_boxes = [
                        {'x': sum(p[0] for p in box) / 4,
                         'y': sum(p[1] for p in box) / 4,
                         'text': text}
                        for box, (text, _) in result_paddle
                    ]
                    if isinstance(table.content, list):
                        for r_idx, row in enumerate(table.content):
                            if not isinstance(row, list):
                                continue
                            for c_idx, cell in enumerate(row):
                                if str(cell.value).strip():
                                    continue
                                x1, y1, x2, y2 = map(int, cell.bbox)
                                texts = [t['text'] for t in ocr_boxes
                                         if x1 <= t['x'] <= x2 and y1 <= t['y'] <= y2]
                                df.iat[r_idx, c_idx] = " ".join(texts)

                    out_csv = output_dir / f"{fname}_img2table_{idx}.csv"
                    df.to_csv(str(out_csv), index=False, header=False, sep=";")
                    print(f"   Zapisano: {out_csv}")

            else:
                print(f"  Słaby (score={score:.2f}) — Paddle fallback")
                grid = paddle_grid_group(result_paddle)
                out_csv = output_dir / f"{fname}_paddle.csv"
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerows(grid)
                print(f"   Zapisano: {out_csv}")
