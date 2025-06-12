#!/usr/bin/env python3
"""
ocr_wzory.py — przetwarza wycięte wzory LaTeX w strukturze:
  ksiazki/<book>/detekcje/wzory/*.png
i zapisuje wyniki do:
  ksiazki/<book>/rezultaty/wzory/latex_wzory.json
  ksiazki/<book>/rezultaty/wzory/wzory.tex
"""
import json
import re
import cv2
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pix2text import Pix2Text

# Konfiguracja katalogów głównych
BOOKS_DIR = Path("ksiazki")

# Model Pix2Text
DEVICE = "cuda:0"
MODEL = "mfr"
p2t = Pix2Text(model=MODEL, device=DEVICE)

# Wzorki do rozpoznania jako formuły
MATH_CHARS = re.compile(r"[A-Za-z0-9\\+\-*/=^_]")
def looks_like_formula(txt: str) -> bool:
    txt = txt.strip("$").strip()
    return bool(MATH_CHARS.search(txt)) and len(txt) >= 3

if __name__ == "__main__":
    # Iteruj po każdej książce
    for book_dir in sorted(BOOKS_DIR.iterdir()):
        if not book_dir.is_dir():
            continue
        src_dir = book_dir / "detekcje" / "wzory"
        out_dir = book_dir / "rezultaty" / "wzory"
        if not src_dir.exists():
            print(f"Brak katalogu detekcji wzorów: {src_dir}, pomijam {book_dir.name}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        # Przechodzimy przez każdy obraz z wzorem
        for img_path in tqdm(sorted(src_dir.glob("*.png")), desc=f"Wzory: {book_dir.name}"):
            # Preprocessing
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, bin_img = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
            rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(rgb)

            # OCR LaTeX
            latex = p2t.recognize(pil_img).strip("$").strip()
            if not looks_like_formula(latex):
                print(f" Usuwam {img_path.name} (nie wzór)")
                img_path.unlink()
                continue
            results[img_path.name] = latex

        # Zapis JSON w out_dir
        json_path = out_dir / "latex_wzory.json"
        json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Zapisano JSON: {json_path}")

        # Zapis TeX w out_dir
        tex_path = out_dir / "wzory.tex"
        with tex_path.open("w", encoding="utf-8") as f:
            f.write(r"""\documentclass{article}
\usepackage{amsmath}
\begin{document}

""")
            for name, latex in results.items():
                f.write(f"% --- {name} ---\n\\[\n{latex}\n\\]\n\n")
            f.write(r"\end{document}")
        print(f"Zapisano TeX: {tex_path}\n")
