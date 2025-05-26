
"""
Połączony skrypt: najpierw uruchamia detekcję i wycinanie (detekcja_elementow.py),
a następnie robi OCR na wygenerowanych _result.png i zapisuje k.txt.


"""

import sys
import os
import re
import subprocess
from pathlib import Path
import glob

import pytesseract
from PIL import Image

# OCR
TESS_CMD  = "/usr/bin/tesseract"         # dostosuj cieżkę
TESS_CFG  = "--oem 1 --psm 3 -l pol"
SUFFIX_RE = re.compile(r"_page0*(\d+)_result\.png$", re.I)

pytesseract.pytesseract.tesseract_cmd = TESS_CMD


def ocr_page(img_path: Path) -> str:
    txt = pytesseract.image_to_string(Image.open(img_path), config=TESS_CFG)

    fixed_lines = []
    for line in txt.splitlines():
        raw = line.strip().lower()

        if any(folder in raw for folder in ("figury", "tabele", "wzory")):
            # 1. usuń śmieci - tylko a-z, 0-9, / _ . dozwolone
            cleaned = re.sub(r"[^a-z0-9_./]", "", raw)

            # 2. poprawy błędów OCR:
            cleaned = cleaned.replace(".page", "_page")        # np. k1.page1 → k1_page1
            cleaned = re.sub(r"\._", "_", cleaned)             # np. _._figure → _figure
            cleaned = re.sub(r"\.(?=figure|table|formula)", "_", cleaned)  # .figure → _figure
            cleaned = re.sub(r"\.(png.*)$", r".png", cleaned)  # .png1 → .png

            # 3. literówki typu figurea / figurei → figure1
            cleaned = re.sub(r"(figure|table|formula)[a-z]{1,2}\.png", r"\1_1.png", cleaned)

            # 4. jeśli wygląda OK
            if cleaned.endswith(".png") and "/" in cleaned:
                line = f"<image/{cleaned}>"
            else:
                line = cleaned

        fixed_lines.append(line)

    return "\n".join(fixed_lines).strip()





def ocr_book(book_dir: Path):
   #Robi OCR wszystkich *_result.png w katalogu i zapisuje kx.txt.
    pngs = sorted(
        book_dir.glob("*_result.png"),
        key=lambda p: int(SUFFIX_RE.search(p.name).group(1))
                     if SUFFIX_RE.search(p.name) else 0
    )
    lines = []
    for idx, png in enumerate(pngs, 1):
        text = ocr_page(png)
        print(f"[{book_dir.name}] strona {idx}: {text[:60]!r}")
        if text:
            lines.append(text)

    if not lines:
        print(f"  Brak rozpoznanego tekstu w {book_dir.name}, pomijam.\n")
        return

    out_txt = book_dir / (book_dir.name.replace("_przetwarzanie", "") + ".txt")
    out_txt.write_text("\n\n".join(lines), encoding="utf-8")
    print(f" Zapisano OCR {out_txt}\n")


def ocr_all(root: Path):
    """Iteruje po wszystkich 'kX_przetwarzanie' w root i odpala ocr_book."""
    if not root.is_dir():
        print(f"Folder nie istnieje: {root}")
        sys.exit(1)

    books = sorted([d for d in root.iterdir() if d.is_dir()])
    if not books:
        print(f"  Nie znaleziono żadnych katalogów z książkami w {root}")
        return

    for b in books:
        det_text = b / "detekcje" / "tekst"
        if not det_text.exists():
            print(f"  Brak detekcji w {det_text}, pomijam.")
            continue

        # przygotuj output
        out_text = b / "rezultaty" / "tekst"
        out_text.mkdir(parents=True, exist_ok=True)

        # pobierz i posortuj pliki png
        pngs = sorted(det_text.glob("*_result.png"),
                      key=lambda p: int(SUFFIX_RE.search(p.name).group(1)))
        lines = []
        for idx, png in enumerate(pngs, 1):
            txt = ocr_page(png)
            print(f"[{b.name}] strona {idx}: {txt[:60]!r}")
            if txt:
                lines.append(txt)
            # opcjonalnie: zapis pojedynczy
            (out_text / f"{b.name}_page{idx}.txt")\
                 .write_text(txt, encoding="utf-8")

        # scalony plik
        if lines:
            merged = "\n\n".join(lines)
            (b / "rezultaty" / f"{b.name}.txt")\
                .write_text(merged, encoding="utf-8")
            print(f"  Zapisano OCR: {b/'rezultaty'/f'{b.name}.txt'}\n")
        else:
            print("  Nic do zapisania\n")

if __name__ == "__main__":

    root_dir = Path(sys.argv[1] if len(sys.argv) >= 2 else "ksiazki")
    print(" OCR \n")
    ocr_all(root_dir)
