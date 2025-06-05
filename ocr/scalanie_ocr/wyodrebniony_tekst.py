import sys
import re
from pathlib import Path
import pytesseract
from PIL import Image

# OCR
TESS_CMD = "/usr/bin/tesseract"  # dostosuj ścieżkę
TESS_CFG = "--oem 1 --psm 3 -l pol"
SUFFIX_RE = re.compile(r"_page0*(\d+)_result\.png$", re.I)

pytesseract.pytesseract.tesseract_cmd = TESS_CMD


def ocr_page(img_path: Path, prefix: str) -> str:
    pil_img = Image.open(img_path)
    big_img = pil_img.resize((pil_img.width * 4, pil_img.height * 4), Image.LANCZOS)
    txt = pytesseract.image_to_string(big_img, config=TESS_CFG) 

    fixed_lines = []
    for line in txt.splitlines():
        raw = line.strip().lower()

        if any(folder in raw for folder in ("figury", "tabele", "wzory")):
            cleaned = re.sub(r"[^a-z0-9_./]", "", raw)

            # OCR błędy: .page → _page, .figure → _figure, itd.
            cleaned = cleaned.replace(".page", "_page")
            cleaned = re.sub(r"\._", "_", cleaned)
            cleaned = re.sub(r"\.(?=figure|table|formula)", "_", cleaned)
            cleaned = re.sub(r"\.(png.*)$", r".png", cleaned)

            # Dodaj brakujące podkreślenia: page5figure → page5_figure
            cleaned = re.sub(r"(page\d+)(figure|table|formula)", r"\1_\2", cleaned)

            # Literówki: igure → figure, able → table, ormula → formula
            cleaned = re.sub(r"_igure", "_figure", cleaned)
            cleaned = re.sub(r"_able", "_table", cleaned)
            cleaned = re.sub(r"_ormula", "_formula", cleaned)

            # Podwójne podkreślenia → pojedyncze
            cleaned = re.sub(r"__+", "_", cleaned)

            # Napraw końcówkę: formulaabc → formula1, formula7xyz → formula7
            match = re.search(r"(figure|table|formula)([^0-9]*)(\d*)", cleaned)
            if match:
                typ = match.group(1)
                num = match.group(3) if match.group(3) else "1"
                cleaned = re.sub(r"(figure|table|formula)[^/]*$", f"{typ}{num}", cleaned)

            # Dopisz .png jeśli nie ma
            if not cleaned.endswith(".png"):
                cleaned += ".png"

            filename = cleaned.split("/")[-1]

            # Usuń błędne podwójne prefixy: k1_ki_ → usuń, potem usuń stare kX_

            filename = re.sub(r"^(k[a-z0-9]+_){1,3}", "", filename)


            # Dodaj poprawny prefix
            filename = f"{prefix}_{filename}"

            # Wybór folderu
            if "figure" in filename:
                folder = "figury"
            elif "table" in filename:
                folder = "tabele"
            elif "formula" in filename:
                folder = "wzory"
            else:
                folder = "inne"

            line = f"<image/{folder}/{filename}>"
        fixed_lines.append(line)

    return "\n".join(fixed_lines).strip()


def ocr_book(book_dir: Path):
    pngs = sorted(
        book_dir.glob("*_result.png"),
        key=lambda p: int(SUFFIX_RE.search(p.name).group(1))
        if SUFFIX_RE.search(p.name) else 0
    )
    prefix = book_dir.name.split('_')[0]
    lines = []
    for idx, png in enumerate(pngs, 1):
        text = ocr_page(png, prefix)
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

        out_text = b / "rezultaty" / "tekst"
        out_text.mkdir(parents=True, exist_ok=True)

        prefix = b.name.split('_')[0]
        pngs = sorted(det_text.glob("*_result.png"),
                      key=lambda p: int(SUFFIX_RE.search(p.name).group(1)))
        lines = []
        for idx, png in enumerate(pngs, 1):
            txt = ocr_page(png, prefix)
            print(f"[{b.name}] strona {idx}: {txt[:60]!r}")
            if txt:
                lines.append(txt)

            (out_text / f"{b.name}_page{idx}.txt").write_text(txt, encoding="utf-8")

        if lines:
            merged = "\n\n".join(lines)
            (b / "rezultaty" / f"{b.name}.txt").write_text(merged, encoding="utf-8")
            print(f"  Zapisano OCR: {b/'rezultaty'/f'{b.name}.txt'}\n")
        else:
            print("  Nic do zapisania\n")


if __name__ == "__main__":
    root_dir = Path(sys.argv[1] if len(sys.argv) >= 2 else "ksiazki")
    print(" OCR \n")
    ocr_all(root_dir)
