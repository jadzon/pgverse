import os, glob, cv2, csv
from paddleocr import PaddleOCR
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR

INPUT= "results/tabele"
OUTPUT= "wyniki_csv"
os.makedirs(OUTPUT, exist_ok=True)
score=0.9
# OCR
ocr_paddle = PaddleOCR(lang="pl", use_angle_cls=True)
ocr_tess = TesseractOCR(lang="pol")

def paddle_score_above_threshold(result, threshold=score):
    good = sum(1 for box in result if box[1][1] >= threshold)
    total = len(result)
    print(f" {good}/{total} boxów ≥ {threshold}")
    return (good / total) if total else 0


def preprocess_for_paddle(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, bin_img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)

def paddle_grid_group(result, y_thresh=20, x_thresh=30):
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
    if current: rows.append(current)
    max_cols = max(len(r) for r in rows)
    final = [ [c['text'] for c in r] + [""]*(max_cols-len(r)) for r in rows ]
    return final

# Przetwarzanie
for path in glob.glob(os.path.join(INPUT, "*.*")):
    fname = os.path.splitext(os.path.basename(path))[0]
    print(f" Przetwarzanie {fname}")

    #  Paddle do oceny jakości
    image_np = preprocess_for_paddle(path)
    result_paddle = ocr_paddle.ocr(image_np, cls=False)[0]
    paddle_score = paddle_score_above_threshold(result_paddle)

    if paddle_score >= score:
        print(f" img2table (score={paddle_score:.2f})")
        doc = Img2TableImage(path, detect_rotation=True)
        tables = doc.extract_tables(ocr=ocr_tess)

        if not tables:
            print(" Brak tabel.")
            continue

        # Przetwarzanie każdej tabeli
        for idx, table in enumerate(tables, 1):
            df = table.df.copy()

            #  Paddle jako uzupełnienie
            ocr_boxes = []
            for box, (text, _) in result_paddle:
                x_c = sum(p[0] for p in box) / 4
                y_c = sum(p[1] for p in box) / 4
                ocr_boxes.append({'x': x_c, 'y': y_c, 'text': text})

            if isinstance(table.content, list):
                for r_idx, row in enumerate(table.content):
                    if not isinstance(row, list): continue
                    for c_idx, cell in enumerate(row):
                        if str(cell.value).strip(): continue
                        x1, y1, x2, y2 = map(int, cell.bbox)
                        texts = [t['text'] for t in ocr_boxes if x1 <= t['x'] <= x2 and y1 <= t['y'] <= y2]
                        df.iat[r_idx, c_idx] = " ".join(texts)

            out_csv = os.path.join(OUTPUT, f"{fname}_img2table.csv")
            df.to_csv(out_csv, index=False, header=False, sep=";")
            print(f" Zapisano: {out_csv}")
    else:
        print(f" Słaby  (score={paddle_score:.2f}) — Paddle")
        table = paddle_grid_group(result_paddle)
        out_csv = os.path.join(OUTPUT, f"{fname}_paddle.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerows(table)
        print(f" Zapisano: {out_csv}")


