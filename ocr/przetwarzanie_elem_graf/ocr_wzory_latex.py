import json, pathlib, re, cv2
from PIL import Image
from tqdm import tqdm
from pix2text import Pix2Text

# sciezki
SRC_DIR    = pathlib.Path("results/wzory")
OUT_JSON   = pathlib.Path("results/latex_wozry.json")
MASTER_TEX = pathlib.Path("results/wzory.tex")

#model
DEVICE = "cuda:0"
MODEL  = "mfr"
p2t    = Pix2Text(model=MODEL, device=DEVICE)

# dodatkowa pomocniczna do sprawdzenia czy na pewno jest to wzor
MATH_CHARS = re.compile(r"[A-Za-z0-9\\+\-*/=^_]")

def looks_like_formula(txt: str) -> bool:
    txt = txt.strip("$").strip()
    return bool(MATH_CHARS.search(txt)) and len(txt) >= 3

# preproces
def preprocess_for_paddle(path: str):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bin_img = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

#glowna petla
results = {}
for img_path in tqdm(sorted(SRC_DIR.iterdir())):
    if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        continue

    latex = p2t.recognize(preprocess_for_paddle(str(img_path))).strip("$").strip()

    if not looks_like_formula(latex):
        print(f" usuwamy {img_path.name} (brak wzoru)")
        img_path.unlink()
        continue

    results[img_path.name] = latex

# .tex // format dokumentu buduje
with MASTER_TEX.open("w", encoding="utf-8") as f:
    f.write(r"""\documentclass{article}
\usepackage{amsmath}
\begin{document}

""")
    for name, latex in results.items():
        f.write(f"% --- {name} ---\n\\[\n{latex}\n\\]\n\n")
    f.write(r"\end{document}")

# zapis do json
OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\n✅ Zapisano: {MASTER_TEX}\n")
