import cv2
import numpy as np
import re
from pathlib import Path
from PIL import Image
from pix2text import Pix2Text
import onnxruntime as ort

print("ONNX Runtime device:", ort.get_device())
print("Providers:", ort.get_all_providers())

# Stałe
IMAGE_PATH = r"D:\nauka\pgverse\pgverse\ocr\tabele_test\1.1.png"
SCALE = 15
EMPTY_THRESH = 50

# Funkcje pomocnicze
def filter_long_lines(bin_img, axis='horizontal', min_frac=0.5):
    H, W = bin_img.shape
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bin_img)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if axis=='horizontal' and w > W * min_frac:
            cv2.drawContours(mask, [c], -1, 255, thickness=-1)
        elif axis=='vertical' and h > H * min_frac:
            cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    return mask


def detect_lines(img, axis='horizontal'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if axis == 'horizontal':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, img.shape[1]//SCALE), 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, img.shape[0]//SCALE)))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)


def extract_cells(img, horiz, vert, eps=20):
    inter = cv2.bitwise_and(horiz, vert)
    ys, xs = np.where(inter > 0)
    def cluster(vals):
        clusters = []
        for v in sorted(vals):
            if not clusters or abs(v - clusters[-1][0]) > eps:
                clusters.append([v])
            else:
                clusters[-1].append(v)
        return [int(np.mean(c)) for c in clusters]
    xs_u = [0] + cluster(xs) + [img.shape[1]]
    ys_u = [0] + cluster(ys) + [img.shape[0]]
    cells = []
    for i in range(len(ys_u)-1):
        row = []
        y1, y2 = ys_u[i], ys_u[i+1]
        for j in range(len(xs_u)-1):
            x1, x2 = xs_u[j], xs_u[j+1]
            w, h = x2-x1, y2-y1
            if w<15 or h<15:
                continue
            row.append((x1,y1,w,h))
        if row:
            cells.append(row)
    return cells

# Inicjalizacja Pix2Text
device = "cuda:0"
model = "mfr"
p2t = Pix2Text(model=model, device=device)
MATH_CHARS = re.compile(r"[A-Za-z0-9\\+\-\*/=^_]" )
def looks_like_formula(txt: str) -> bool:
    txt = txt.strip("$").strip()
    return bool(MATH_CHARS.search(txt)) and len(txt) >= 2

if __name__ == '__main__':
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Nie można wczytać obrazu: {IMAGE_PATH}")

    # 1) detekcja i filtracja linii
    horiz = filter_long_lines(detect_lines(img, 'horizontal'), 'horizontal', min_frac=0.5)
    vert  = filter_long_lines(detect_lines(img, 'vertical'),   'vertical',   min_frac=0.5)

    # 2) ekstrakcja komórek
    cells = extract_cells(img, horiz, vert)

    # 3) OCR i formatowanie zawartości
    table = []
    for row in cells:
        row_tex = []
        for (x,y,w,h) in row:
            crop = img[y:y+h, x:x+w]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
            pil = Image.fromarray(cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB))
            txt = p2t.recognize(pil).strip("$").strip()
            if looks_like_formula(txt):
                cell = f"${txt}$"
            else:
                cell = f"\\text{{{txt}}}" if txt else ""
            row_tex.append(cell)
        table.append(row_tex)

    # 4) usuwanie niepotrzebnych kolumn (numeracja i puste ramki)
        # 4) usuwanie pustych kolumn generowanych przez krawędzie
    if table:
        # Transponujemy, by móc łatwo sprawdzić każdą kolumnę
        cols = list(zip(*table))
        # Zobaczmy, czy pierwsza kolumna jest całkowicie pusta
        first_col = cols[0]
        if all(cell.strip()=="" for cell in first_col):
            # jeżeli pusta – usuń ją
            table = [row[1:] for row in table]
            cols = cols[1:]
        # Teraz zlokalizujmy pierwszy i ostatni niepusty indeks w pozostałych kolumnach
        first_idx, last_idx = None, None
        for idx, col in enumerate(cols):
            if any(cell.strip() for cell in col):
                if first_idx is None:
                    first_idx = idx
                last_idx = idx
        if first_idx is None:
            ncols = 0
        else:
            # Przytnijmy obie skrajne puste kolumny
            # Uwaga: jeśli odrzuciliśmy pierwszą, musimy odsłonić offset
            offset = 1 if all(cell.strip()=="" for cell in first_col) else 0
            trimmed = [row[offset+first_idx : offset+last_idx+1] for row in table]
            table = trimmed
            ncols = last_idx - first_idx + 1
    else:
        ncols = 0


    # 5) generacja LaTeX
    tex = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath,array}",
        r"\begin{document}",
        "",
        rf"\begin{{tabular}}{{|{'c|'*ncols}}}",
        r"\hline"
    ]
    for r in table:
        tex.append(" & ".join(r) + r" \\ \hline")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{document}")

    out = Path(IMAGE_PATH).with_suffix(".tex")
    out.write_text("\n".join(tex), encoding="utf-8")
    print(f"Zapisano LaTeX: {out}")
