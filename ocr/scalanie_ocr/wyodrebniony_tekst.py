
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


def run_detection():
    """
    Uruchamia skrypt detekcji, który powinien wygenerować:
      results/kX_przetwarzanie/.../*_result.png
'detekcja_elementow.py' musi  lezec w tym samym folderze co ten plik.
    """
    this = Path(__file__).resolve()
    det_script = this.parent / "detekcja_elementow.py"
    if not det_script.exists():
        print(f" Nie znaleziono skryptu : {det_script}")
        sys.exit(1)
    print(f" Uruchamiam : {det_script.name} …")
    # to samo środowisko Python
    subprocess.check_call([sys.executable, str(det_script)])
    print("1 czesc zakończona.\n")


def ocr_page(img_path: Path) -> str:

    txt = pytesseract.image_to_string(Image.open(img_path), config=TESS_CFG)
    return txt.strip()


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

    books = sorted(root.glob("k*_przetwarzanie"))
    if not books:
        print(f"  Nie znaleziono katalogów k*_przetwarzanie w {root}")
        return

    for b in books:
        ocr_book(b)


if __name__ == "__main__":
    # 1
    run_detection()

    # 2) oCR w katalogu wyników 'results'
    root_dir = Path(sys.argv[1] if len(sys.argv) >= 2 else "results")
    print(" OCR \n")
    ocr_all(root_dir)
