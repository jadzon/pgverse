import cv2
import numpy as np
#analiza tabel zostanie zrobiona na podstawie detekcji linii, z tego względu, że gotowe rozwiązania (np paddle) nie działają wystarczająco dobrze
# i trzeba im trochę pomóc
# w tym celu wykorzystamy morfologię matematyczną do detekcji linii poziomych i pionowych
# narazie program wykrywa tylko linie i wycina poszczególne komórki
# i wyświetla je w oknie
# Ścieżka do obrazu wpisana na stałe
IMAGE_PATH = r"D:\nauka\pgverse\pgverse\ocr\tabele_test\1.png"
# Parametr skalowania dla detekcji linii
SCALE = 15
# Minimalna liczba pikseli niebędących tłem, aby uznać komórkę za niepustą
EMPTY_THRESH = 50

def detect_lines(img, axis='horizontal'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if axis == 'horizontal':
        size = max(1, img.shape[1] // SCALE)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
    else:
        size = max(1, img.shape[0] // SCALE)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)


def extract_cells(img, horiz, vert):
    inter = cv2.bitwise_and(horiz, vert)
    ys, xs = np.where(inter > 0)
    def cluster(vals, eps=10):
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
    for i in range(len(ys_u) - 1):
        row = []
        y1, y2 = ys_u[i], ys_u[i+1]
        for j in range(len(xs_u) - 1):
            x1, x2 = xs_u[j], xs_u[j+1]
            w, h = x2 - x1, y2 - y1
            # pomiń bardzo małe obszary
            if w < 5 or h < 5:
                continue
            row.append((x1, y1, w, h))
        if row:
            cells.append(row)
    return cells

if __name__ == '__main__':
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Nie można wczytać obrazu: {IMAGE_PATH}")
    horiz = detect_lines(img, 'horizontal')
    vert = detect_lines(img, 'vertical')
    # Wyświetlenie nakładki
    red = np.zeros_like(img)
    red[horiz>0] = (0,0,255)
    red[vert>0] = (0,0,255)
    overlay = cv2.addWeighted(img,1,red,1,0)
    cv2.imshow('Red Lines Overlay', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Podział na komórki
    cells = extract_cells(img, horiz, vert)
    # Wyświetlanie regionów z pominięciem pustych
    for i, row in enumerate(cells):
        for j, (x,y,w,h) in enumerate(row):
            crop = img[y:y+h, x:x+w]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
            nonzero = cv2.countNonZero(bw)
            if nonzero < EMPTY_THRESH:
                print(f"Pomijam pustą komórkę [{i},{j}] (pikseli: {nonzero})")
                continue
            cv2.imshow(f'Cell [{i},{j}]', crop)
            cv2.waitKey(0)
            cv2.destroyWindow(f'Cell [{i},{j}]')
    cv2.destroyAllWindows()
