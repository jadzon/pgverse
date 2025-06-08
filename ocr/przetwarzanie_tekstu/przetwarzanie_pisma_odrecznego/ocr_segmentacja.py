import fitz
import cv2
import numpy as np
import os
import argparse
from PIL import Image
import io

def segment_page_lines(img, min_area=500):
    """
    Segmentuje obraz tak, żeby każdy kontur odpowiadał jednej linii tekstu.
    - img: wejściowy obraz BGR
    - min_area: minimalna powierzchnia konturu, by go uwzględnić
    Zwraca listę boxów (x, y, w, h) posortowanych podług y.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Inwersja + binarizacja
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dylatacja pozioma: kernel szeroki i niski, by łączyć znaki w jednej linii
    img_h, img_w = bw.shape
    # szerokość kernelu to np. 20% szerokości obrazu, wysokość 1
    k_w = max(1, img_w // 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    dilated = cv2.dilate(bw, kernel, iterations=1)

    # Znajdź kontury
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            # Rozszerzamy lekko box w pionie, by złapać np. części liter
            pad = int(0.1 * h)
            y0 = max(0, y - pad)
            y1 = min(img_h, y + h + pad)
            boxes.append((x, y0, w, y1 - y0))
    # Sortuj od góry do dołu
    boxes.sort(key=lambda b: b[1])
    return boxes

def segment_pdf_lines(input_pdf, output_dir, min_area=500, dpi=200):
    """
    Dla każdej strony PDF renderuje obraz, segmentuje na linijki tekstu
    i zapisuje każdy wiersz jako osobny obraz PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(input_pdf)

    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        boxes = segment_page_lines(img, min_area=min_area)

        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            line_img = img[y:y+h, x:x+w]
            fname = f"page{page_num:02d}_line{idx:03d}.png"
            cv2.imwrite(os.path.join(output_dir, fname), line_img)

        print(f"Strona {page_num:02d}: wycięto {len(boxes)} linii.")

    print("Segmentacja zakończona.", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segmentacja PDF na pojedyncze wiersze tekstu"
    )
    parser.add_argument("input_pdf", help="Wejściowy plik PDF (np. n1.pdf)")
    parser.add_argument("output_folder", help="Folder na wycinki linijek")
    parser.add_argument(
        "--min_area",
        type=int,
        default=500,
        help="Minimalna powierzchnia linii w px (domyślnie 500)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rozdzielczość renderingu stron (domyślnie 200)",
    )
    args = parser.parse_args()

    segment_pdf_lines(args.input_pdf, args.output_folder, args.min_area, args.dpi)

