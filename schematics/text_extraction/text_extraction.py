import cv2
import numpy as np
import os
import easyocr
import time
import argparse
import pytesseract
from imutils.object_detection import non_max_suppression
from pathlib import Path

# Dodajemy zmienną określającą ścieżkę do katalogu projektu
PROJEKT_DIR = "."  # Używamy bieżącego katalogu jako punkt odniesienia

def wczytaj_przetworzone_obrazy(folder_przetworzone, typ_obrazu="binaryzacja_ulepszona"):
    """
    Wczytuje przetworzone obrazy z określonego folderu.
    
    Args:
        folder_przetworzone: Ścieżka do folderu z przetworzonymi obrazami
        typ_obrazu: Typ przetworzenia do wczytania (np. "binaryzacja", "binaryzacja_ulepszona")
        
    Returns:
        Lista krotek (nazwa_pliku, obraz)
    """
    obrazy = []
    
    # Pełna ścieżka do folderu
    pelna_sciezka = os.path.join(PROJEKT_DIR, folder_przetworzone)
    
    # Szukamy wszystkich plików dla danego typu przetworzenia
    wzorzec = f"*_{typ_obrazu}.png"
    for plik in Path(pelna_sciezka).glob(wzorzec):
        obraz = cv2.imread(str(plik))
        if obraz is not None:
            # Pobieramy podstawową nazwę pliku (bez _typ_obrazu.png)
            nazwa_bazowa = os.path.basename(plik)
            nazwa_bazowa = nazwa_bazowa.replace(f"_{typ_obrazu}.png", "")
            obrazy.append((nazwa_bazowa, obraz))
    
    print(f"Wczytano {len(obrazy)} przetworzonych obrazów typu {typ_obrazu}")
    return obrazy

def wczytaj_przetworzenia_obrazu(folder_przetworzone, nazwa_bazowa):
    """
    Wczytuje wszystkie dostępne przetworzenia dla pojedynczego obrazu.
    
    Args:
        folder_przetworzone: Ścieżka do folderu z przetworzonymi obrazami
        nazwa_bazowa: Podstawowa nazwa obrazu
        
    Returns:
        Słownik {typ_przetworzenia: obraz}
    """
    przetworzone = {}
    
    # Pełna ścieżka do folderu
    pelna_sciezka_folderu = os.path.join(PROJEKT_DIR, folder_przetworzone)
    
    # Typowe przetworzenia dostępne w text_preprocesing.py
    typy_przetworzen = [
        "szary", "szary_wyrownany", "szary_odszumiony", 
        "binaryzacja", "binaryzacja_ulepszona", "krawedzie"
    ]
    
    for typ in typy_przetworzen:
        sciezka = os.path.join(pelna_sciezka_folderu, f"{nazwa_bazowa}_{typ}.png")
        if os.path.exists(sciezka):
            obraz = cv2.imread(sciezka)
            if obraz is not None:
                # Jeśli obraz jest kolorowy ale powinien być w skali szarości, konwertujemy
                if typ != "oryginalny" and len(obraz.shape) == 3:
                    obraz = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
                przetworzone[typ] = obraz
    
    return przetworzone

def wczytaj_oryginalne_obrazy(sciezki_folderow, nazwy_bazowe):
    """
    Wczytuje oryginalne wersje obrazów na podstawie listy nazw bazowych.
    
    Args:
        sciezki_folderow: Lista ścieżek do folderów z oryginalnymi obrazami
        nazwy_bazowe: Lista nazw bazowych obrazów
        
    Returns:
        Słownik {nazwa_bazowa: oryginalny_obraz}
    """
    oryginalne_obrazy = {}
    
    for nazwa in nazwy_bazowe:
        for folder in sciezki_folderow:
            # Pełna ścieżka do folderu
            pelna_sciezka_folderu = os.path.join(PROJEKT_DIR, folder)
            
            for rozszerzenie in [".jpg", ".jpeg", ".png"]:
                potencjalna_sciezka = os.path.join(pelna_sciezka_folderu, nazwa + rozszerzenie)
                if os.path.exists(potencjalna_sciezka):
                    obraz = cv2.imread(potencjalna_sciezka)
                    if obraz is not None:
                        oryginalne_obrazy[nazwa] = obraz
                        break
    
    print(f"Wczytano {len(oryginalne_obrazy)} oryginalnych obrazów")
    return oryginalne_obrazy

def preprocessing_dla_ocr(obraz):
    """
    Wykonuje specjalny preprocessing obrazu zoptymalizowany dla OCR.
    
    Args:
        obraz: Obraz w formacie OpenCV (numpy array)
        
    Returns:
        Przetworzony obraz
    """
    # Upewniamy się, że obraz jest w skali szarości
    if len(obraz.shape) == 3:
        szary = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
    else:
        szary = obraz.copy()
    
    # Umiarkowane wyrównanie histogramu dla małych napisów
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    szary_wyrownany = clahe.apply(szary)
    
    # Odszumianie z zachowaniem krawędzi
    szary_odszumiony = cv2.fastNlMeansDenoising(szary_wyrownany, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Binaryzacja adaptacyjna
    binaryzacja = cv2.adaptiveThreshold(
        szary_odszumiony, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Operacje morfologiczne dla poprawy czytelności małych napisów
    kernel = np.ones((2, 2), np.uint8)
    binaryzacja_ulepszona = cv2.morphologyEx(binaryzacja, cv2.MORPH_CLOSE, kernel)
    
    # Odwracamy obraz jeśli to konieczne (oczekujemy czarnego tekstu na białym tle)
    if np.mean(binaryzacja_ulepszona) > 127:
        binaryzacja_ulepszona = cv2.bitwise_not(binaryzacja_ulepszona)
    
    return binaryzacja_ulepszona

def wykryj_obszary_tekstu_mser(obraz, min_area=10, max_area=2000):
    """
    Wykrywa potencjalne obszary tekstu za pomocą algorytmu MSER.
    
    Args:
        obraz: Obraz w formacie OpenCV (numpy array)
        min_area: Minimalna powierzchnia obszaru
        max_area: Maksymalna powierzchnia obszaru
        
    Returns:
        Lista regionów (x, y, w, h)
    """
    # Upewniamy się, że obraz jest w skali szarości
    if len(obraz.shape) == 3:
        szary = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
    else:
        szary = obraz.copy()
    
    # Inicjalizacja detektora MSER
    mser = cv2.MSER_create(
        _min_area=min_area,
        _max_area=max_area,
        _delta=5,
        _max_variation=0.5
    )
    
    # Wykrywanie regionów
    regiony, _ = mser.detectRegions(szary)
    
    # Konwersja regionów na prostokąty
    prostokaty = []
    for region in regiony:
        x, y, w, h = cv2.boundingRect(region)
        # Filtrowanie regionów o niewłaściwych proporcjach
        if 0.1 < w/h < 10 and w > 5 and h > 5:
            prostokaty.append((x, y, w, h))
    
    # Łączenie zachodzących na siebie prostokątów
    if prostokaty:
        prostokaty = non_max_suppression(np.array(prostokaty), 0.4)
        
    return prostokaty

def wykryj_obszary_tekstu_east(obraz, min_confidence=0.5):
    """
    Wykrywa potencjalne obszary tekstu za pomocą modelu EAST.
    
    Args:
        obraz: Obraz w formacie OpenCV (numpy array)
        min_confidence: Minimalny poziom pewności detekcji
        
    Returns:
        Lista regionów (x, y, w, h)
    """
    # Sprawdzamy, czy model EAST jest dostępny
    model_path = os.path.join(PROJEKT_DIR, "frozen_east_text_detection.pb")
    if not os.path.exists(model_path):
        print(f"Brak modelu EAST w ścieżce {model_path}")
        return []
    
    # Przygotowanie obrazu dla modelu EAST
    (H, W) = obraz.shape[:2]
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    
    if len(obraz.shape) == 2:  # Obraz w skali szarości
        obraz_rgb = cv2.cvtColor(obraz, cv2.COLOR_GRAY2RGB)
    else:
        obraz_rgb = obraz.copy()
    
    # Przeskalowanie obrazu
    obraz_east = cv2.resize(obraz_rgb, (newW, newH))
    blob = cv2.dnn.blobFromImage(obraz_east, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    
    # Wczytanie modelu EAST
    net = cv2.dnn.readNet(model_path)
    
    # Ustawienie nazw warstw wyjściowych
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    
    # Propagacja obrazu przez sieć
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    
    # Dekodowanie wyników
    (rects, confidences) = dekoduj_przewidywania_east(scores, geometry, min_confidence)
    
    # Stosowanie non-maxima suppression do usunięcia nakładających się bounding boxów
    if len(rects) > 0:
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        
        # Przeskalowanie bounding boxów do oryginalnych wymiarów
        prostokaty = []
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            prostokaty.append((startX, startY, endX - startX, endY - startY))
        
        return prostokaty
    
    return []

def dekoduj_przewidywania_east(scores, geometry, min_confidence):
    """
    Dekoduje wyniki modelu EAST.
    
    Args:
        scores: Mapa pewności z modelu
        geometry: Geometria bounding boxów
        min_confidence: Minimalny poziom pewności
    
    Returns:
        Tuple list prostokątów i odpowiadających im poziomów pewności
    """
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def wykryj_tekst_easyocr(obraz, reader, jezyk='pl', min_confidence=0.2):
    """
    Wykrywa tekst na obrazie za pomocą EasyOCR z ulepszoną preobróbką.
    
    Args:
        obraz: Obraz w formacie OpenCV (numpy array)
        reader: Załadowany model EasyOCR
        jezyk: Kod języka do detekcji (domyślnie polski)
        min_confidence: Minimalny poziom pewności detekcji
        
    Returns:
        Lista wykrytych tekstów z pozycjami i prawdopodobieństwami
    """
    # Zastosowanie wielopoziomowego preprocessingu
    przetworzone_wersje = wielopoziomowy_preprocessing(obraz)
    
    najlepsze_wyniki = []
    najlepsza_pewnosc_srednia = 0
    
    # Sprawdzamy różne wersje przetworzenia
    for nazwa_przetworzenia, obraz_przetworzony in przetworzone_wersje.items():
        # Konwersja do RGB dla EasyOCR
        if len(obraz_przetworzony.shape) == 2:
            img_rgb = cv2.cvtColor(obraz_przetworzony, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(obraz_przetworzony, cv2.COLOR_BGR2RGB)
        
        # Parametry dla lepszej detekcji
        wyniki = reader.readtext(
            img_rgb,
            detail=1,
            paragraph=False,
            decoder='beamsearch',
            beamWidth=5,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=min_confidence,
            link_threshold=0.3,
            mag_ratio=1.5
        )
        
        # Filtrowanie wyników o niskim poziomie pewności
        wyniki_filtrowane = [wynik for wynik in wyniki if wynik[2] >= min_confidence]
        
        # Wybieramy przetworzenie z najlepszą średnią pewnością
        if wyniki_filtrowane:
            pewnosc_srednia = sum(wynik[2] for wynik in wyniki_filtrowane) / len(wyniki_filtrowane)
            if pewnosc_srednia > najlepsza_pewnosc_srednia:
                najlepsza_pewnosc_srednia = pewnosc_srednia
                najlepsze_wyniki = wyniki_filtrowane
    
    return najlepsze_wyniki

def wykryj_tekst_tesseract(obraz, jezyk='pol', config=''):
    """
    Wykrywa tekst na obrazie za pomocą Tesseract OCR z ulepszoną preobróbką.
    
    Args:
        obraz: Obraz w formacie OpenCV (numpy array)
        jezyk: Kod języka do detekcji (domyślnie polski)
        config: Dodatkowa konfiguracja dla Tesseract
        
    Returns:
        Lista wykrytych tekstów z pozycjami i prawdopodobieństwami
    """
    try:
        # Predefiniowane konfiguracje dla różnych typów tekstu
        if not config:
            config = '--oem 1 --psm 6 -l {} --dpi 300'.format(jezyk)
        
        # Zastosowanie wielopoziomowego preprocessingu
        przetworzone_wersje = wielopoziomowy_preprocessing(obraz)
        
        najlepsze_wyniki = []
        najlepsza_pewnosc_suma = 0
        
        # Sprawdzamy różne wersje przetworzenia
        for nazwa_przetworzenia, obraz_przetworzony in przetworzone_wersje.items():
            # Wykrywanie tekstu z danymi o pozycji
            try:
                ocr_data = pytesseract.image_to_data(
                    obraz_przetworzony, 
                    lang=jezyk, 
                    config=config, 
                    output_type=pytesseract.Output.DICT
                )
            
                # Przetwarzanie wyników
                wyniki = []
                liczba_wykrytych = len(ocr_data['text'])
                
                for i in range(liczba_wykrytych):
                    # Filtrujemy puste wyniki
                    tekst = ocr_data['text'][i].strip()
                    if not tekst:
                        continue
                    
                    # Pobieramy dane o pewności (0-100%)
                    pewnosc = float(ocr_data['conf'][i]) / 100.0
                    if pewnosc < 0.01:  # Filtrujemy bardzo niskie pewności
                        continue
                    
                    # Pobieramy koordynaty
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    # Tworzymy bounding box w formacie EasyOCR
                    bbox = [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ]
                    
                    wyniki.append((bbox, tekst, pewnosc))
                
                # Wybieramy przetworzenie z najlepszą sumą pewności
                suma_pewnosci = sum(wynik[2] for wynik in wyniki)
                if suma_pewnosci > najlepsza_pewnosc_suma:
                    najlepsza_pewnosc_suma = suma_pewnosci
                    najlepsze_wyniki = wyniki
            
            except Exception as e:
                print(f"Błąd Tesseract dla przetworzenia {nazwa_przetworzenia}: {e}")
                continue
        
        return najlepsze_wyniki
    
    except Exception as e:
        print(f"Błąd podczas używania Tesseract OCR: {e}")
        return []

def polacz_wyniki_ocr(wyniki_easyocr, wyniki_tesseract, iou_threshold=0.5):
    """
    Łączy wyniki z różnych metod OCR, usuwając duplikaty.
    
    Args:
        wyniki_easyocr: Lista wyników z EasyOCR
        wyniki_tesseract: Lista wyników z Tesseract
        iou_threshold: Próg IoU dla uznania dwóch bboxów za ten sam region
        
    Returns:
        Lista połączonych wyników bez duplikatów
    """
    wszystkie_wyniki = []
    
    # Dodajemy najpierw wyniki z EasyOCR
    for bbox, tekst, pewnosc in wyniki_easyocr:
        wszystkie_wyniki.append({
            "bbox": bbox,
            "tekst": tekst,
            "pewnosc": pewnosc,
            "zrodlo": "easyocr"
        })
    
    # Następnie dodajemy wyniki z Tesseract, unikając duplikatów
    for bbox, tekst, pewnosc in wyniki_tesseract:
        is_duplicate = False
        for wynik in wszystkie_wyniki:
            if oblicz_iou(bbox, wynik["bbox"]) > iou_threshold:
                # W przypadku duplikatu, wybieramy wynik o wyższej pewności
                if pewnosc > wynik["pewnosc"]:
                    wynik["tekst"] = tekst
                    wynik["pewnosc"] = pewnosc
                    wynik["zrodlo"] = "tesseract"
                is_duplicate = True
                break
        
        if not is_duplicate:
            wszystkie_wyniki.append({
                "bbox": bbox,
                "tekst": tekst,
                "pewnosc": pewnosc,
                "zrodlo": "tesseract"
            })
    
    # Konwersja z powrotem do formatu (bbox, tekst, pewnosc)
    wyniki_polaczone = [(wynik["bbox"], wynik["tekst"], wynik["pewnosc"]) for wynik in wszystkie_wyniki]
    
    return wyniki_polaczone

def oblicz_iou(bbox1, bbox2):
    """
    Oblicza IoU (Intersection over Union) dla dwóch bounding boxów.
    
    Args:
        bbox1: Pierwszy bounding box w formacie [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        bbox2: Drugi bounding box w tym samym formacie
        
    Returns:
        Wartość IoU (0.0 - 1.0)
    """
    # Konwersja na format [x, y, w, h]
    def get_rect(bbox):
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    rect1 = get_rect(bbox1)
    rect2 = get_rect(bbox2)
    
    # Rozpakowanie koordynatów
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    
    # Obliczenie współrzędnych przecięcia
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Sprawdzenie, czy prostokąty się przecinają
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Obliczenie pola przecięcia
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Obliczenie pól obu prostokątów
    rect1_area = w1 * h1
    rect2_area = w2 * h2
    
    # Obliczenie pola sumy (suma - przecięcie)
    union_area = rect1_area + rect2_area - intersection_area
    
    # Obliczenie IoU
    iou = intersection_area / float(union_area)
    
    return iou

def oznacz_wykryty_tekst(obraz_oryginalny, wykryte_teksty, prog_pewnosci=0.2):
    """
    Oznacza wykryty tekst na oryginalnym obrazie.
    
    Args:
        obraz_oryginalny: Oryginalny obraz
        wykryte_teksty: Lista wykrytych tekstów z pozycjami i prawdopodobieństwami
        prog_pewnosci: Minimalny próg pewności dla wyświetlenia wyniku
        
    Returns:
        Obraz z oznaczeniami
    """
    obraz_z_oznaczeniami = obraz_oryginalny.copy()
    
    for (bbox, text, pewnosc) in wykryte_teksty:
        if pewnosc >= prog_pewnosci:
            # Konwertujemy bounding box do formatu (x, y, w, h)
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            
            # Dobieramy kolor w zależności od pewności (od czerwonego przez żółty do zielonego)
            # Im wyższa pewność, tym bardziej zielony
            r = int(255 * (1 - pewnosc))
            g = int(255 * pewnosc)
            b = 0
            kolor = (b, g, r)  # BGR w OpenCV
            
            # Rysujemy prostokąt
            cv2.line(obraz_z_oznaczeniami, tl, tr, kolor, 2)
            cv2.line(obraz_z_oznaczeniami, tr, br, kolor, 2)
            cv2.line(obraz_z_oznaczeniami, br, bl, kolor, 2)
            cv2.line(obraz_z_oznaczeniami, bl, tl, kolor, 2)
            
            # Przygotowujemy tekst z pewnością
            tekst_do_wyswietlenia = f"{text}: {pewnosc:.2f}"
            
            # Ustawiamy tło pod tekstem
            (text_width, text_height), _ = cv2.getTextSize(
                tekst_do_wyswietlenia, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                obraz_z_oznaczeniami,
                (tl[0], tl[1] - text_height - 10),
                (tl[0] + text_width, tl[1]),
                (0, 0, 0),
                -1
            )
            
            # Umieszczamy tekst nad prostokątem
            cv2.putText(
                obraz_z_oznaczeniami, 
                tekst_do_wyswietlenia, 
                (tl[0], tl[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
    
    return obraz_z_oznaczeniami

def zapisz_wyniki(obrazy_z_oznaczeniami, folder_wyjsciowy="Wyniki_OCR"):
    """
    Zapisuje obrazy z oznaczeniami do podanego folderu, grupując je według folderów źródłowych.
    
    Args:
        obrazy_z_oznaczeniami: Słownik {nazwa_bazowa: obraz_z_oznaczeniami}
        folder_wyjsciowy: Folder do zapisania wyników
    """
    # Pełna ścieżka do folderu wyjściowego
    pelna_sciezka_wyjsciowa = os.path.join(PROJEKT_DIR, folder_wyjsciowy)
    os.makedirs(pelna_sciezka_wyjsciowa, exist_ok=True)
    
    # Grupowanie obrazów według folderów źródłowych
    foldery = {}
    for nazwa, obraz in obrazy_z_oznaczeniami.items():
        folder_zrodlowy = nazwa.split('_')[0]
        if folder_zrodlowy not in foldery:
            foldery[folder_zrodlowy] = {}
            
            # Tworzymy podfolder dla tej kategorii
            os.makedirs(os.path.join(pelna_sciezka_wyjsciowa, folder_zrodlowy), exist_ok=True)
            
        foldery[folder_zrodlowy][nazwa] = obraz
    
    # Zapisujemy obrazy do odpowiednich podfolderów
    for folder, obrazy in foldery.items():
        for nazwa, obraz in obrazy.items():
            nazwa_oryginalna = '_'.join(nazwa.split('_')[1:])  # Usuwamy prefiks folderu
            sciezka_wyjsciowa = os.path.join(pelna_sciezka_wyjsciowa, folder, f"{nazwa_oryginalna}_ocr.png")
            cv2.imwrite(sciezka_wyjsciowa, obraz)
    
    print(f"Zapisano {len(obrazy_z_oznaczeniami)} obrazów z oznaczeniami w folderze {folder_wyjsciowy}")
    for folder, obrazy in foldery.items():
        print(f"  - {folder}: {len(obrazy)} obrazów")

def wielopoziomowy_preprocessing(obraz):
    """
    Wykonuje wielopoziomowe przetwarzanie obrazu, zwracając różne wersje
    dla optymalizacji detekcji tekstu.
    
    Args:
        obraz: Obraz w formacie OpenCV (numpy array)
        
    Returns:
        Słownik różnych przetworzeń obrazu
    """
    # Upewniamy się, że obraz jest w skali szarości
    if len(obraz.shape) == 3:
        szary = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
    else:
        szary = obraz.copy()
    
    # CLAHE dla poprawy kontrastu
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    szary_wyrownany = clahe.apply(szary)
    
    # Odszumianie
    szary_odszumiony = cv2.fastNlMeansDenoising(szary_wyrownany, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Binaryzacja adaptacyjna
    binaryzacja = cv2.adaptiveThreshold(
        szary_odszumiony, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Ulepszona binaryzacja z operacjami morfologicznymi
    kernel = np.ones((2, 2), np.uint8)
    binaryzacja_ulepszona = cv2.morphologyEx(binaryzacja, cv2.MORPH_CLOSE, kernel)
    
    # Gradient Sobel - przydatny dla detekcji krawędzi tekstu
    sobelx = cv2.Sobel(szary_odszumiony, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(szary_odszumiony, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_krawedzie = np.uint8(255 * sobel_combined / np.max(sobel_combined))
    
    return {
        "szary": szary,
        "szary_wyrownany": szary_wyrownany,
        "szary_odszumiony": szary_odszumiony,
        "binaryzacja": binaryzacja,
        "binaryzacja_ulepszona": binaryzacja_ulepszona,
        "krawedzie": sobel_krawedzie
    }

def znajdz_folder(sciezka):
    """
    Znajduje folder niezależnie od wielkości liter.
    
    Args:
        sciezka: Ścieżka do poszukiwanego folderu
        
    Returns:
        Prawidłowa ścieżka lub None jeśli nie znaleziono
    """
    # Budujemy pełną ścieżkę
    if not os.path.isabs(sciezka):
        pelna_sciezka = os.path.join(PROJEKT_DIR, sciezka)
    else:
        pelna_sciezka = sciezka
    
    sciezka_bazowa = os.path.dirname(pelna_sciezka)
    nazwa_folderu = os.path.basename(pelna_sciezka)
    
    if os.path.exists(pelna_sciezka):
        return pelna_sciezka
    
    # Sprawdźmy czy istnieje folder z inną wielkością liter
    if os.path.exists(sciezka_bazowa):
        for element in os.listdir(sciezka_bazowa):
            if element.lower() == nazwa_folderu.lower() and os.path.isdir(os.path.join(sciezka_bazowa, element)):
                print(f"Znaleziono folder {element} zamiast {nazwa_folderu}")
                return os.path.join(sciezka_bazowa, element)
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Ekstrakcja tekstu z przetworzonych obrazów')
    parser.add_argument('--typ', type=str, default='binaryzacja_ulepszona',
                        help='Typ przetworzonego obrazu do użycia (np. binaryzacja, binaryzacja_ulepszona)')
    parser.add_argument('--prog', type=float, default=0.2,
                        help='Minimalny próg pewności dla wyświetlenia wykrytego tekstu (0.0-1.0)')
    parser.add_argument('--jezyk', type=str, default='pol',
                        help='Język do wykrywania (pol, eng, itp.)')
    parser.add_argument('--metoda', type=str, default='hybrid',
                        choices=['easyocr', 'tesseract', 'hybrid'],
                        help='Metoda OCR do użycia (easyocr, tesseract, hybrid)')
    parser.add_argument('--wybor_najlepszego', action='store_true',
                        help='Czy wybrać najlepsze przetworzenie dla każdego obrazu')
    parser.add_argument('--foldery', type=str, default="Dataset/Automatyka,Dataset/Elektroniczne",
                        help='Lista folderów z obrazami, oddzielona przecinkami')
    args = parser.parse_args()
    
    # Folder z przetworzonymi obrazami (ścieżka względna projektu)
    folder_przetworzone = "Dataset/Przetworzone"
    
    # Możliwość określenia folderów przez argument
    sciezki_folderow = args.foldery.split(',')
    
    # Sprawdzenie istnienia folderów
    istniejace_foldery = []
    for sciezka in sciezki_folderow:
        sciezka = sciezka.strip()
        # Budujemy pełną ścieżkę
        pelna_sciezka = os.path.join(PROJEKT_DIR, sciezka)
        
        if os.path.exists(pelna_sciezka) and os.path.isdir(pelna_sciezka):
            istniejace_foldery.append(sciezka)  # Dodajemy ścieżkę względną
            print(f"Folder {sciezka} istnieje i zostanie przetworzony.")
        else:
            print(f"UWAGA: Folder {sciezka} nie istnieje lub nie jest katalogiem!")
            
            # Sprawdź czy to może być problem z wielkością liter
            katalog_nadrzedny = os.path.dirname(pelna_sciezka)
            nazwa_katalogu = os.path.basename(pelna_sciezka)
            if os.path.exists(katalog_nadrzedny):
                for element in os.listdir(katalog_nadrzedny):
                    if element.lower() == nazwa_katalogu.lower() and os.path.isdir(os.path.join(katalog_nadrzedny, element)):
                        # Tworzymy ścieżkę względną
                        poprawna_sciezka_wzgledna = os.path.join(os.path.dirname(sciezka), element)
                        istniejace_foldery.append(poprawna_sciezka_wzgledna)
                        print(f"Znaleziono podobny folder {poprawna_sciezka_wzgledna}, który zostanie przetworzony.")
    
    if not istniejace_foldery:
        print("Nie znaleziono żadnych istniejących folderów do przetworzenia!")
        return
    
    # Czas początku przetwarzania
    czas_start = time.time()
    
    # Wczytujemy obrazy bezpośrednio z oryginalnych folderów
    oryginalne_obrazy = {}
    liczba_obrazow_folder = {}
    
    for folder in istniejace_foldery:
        folder_nazwa = os.path.basename(folder)
        liczba_obrazow_folder[folder_nazwa] = 0
        
        print(f"Wczytuję obrazy z folderu: {folder}")
        pelna_sciezka_folderu = os.path.join(PROJEKT_DIR, folder)
        
        try:
            for plik in os.listdir(pelna_sciezka_folderu):
                pelna_sciezka = os.path.join(pelna_sciezka_folderu, plik)
                if os.path.isfile(pelna_sciezka) and plik.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        nazwa_bez_rozszerzenia = os.path.splitext(plik)[0]
                        
                        # Dodajemy prefix folderu do nazwy obrazu, aby uniknąć kolizji nazw
                        nazwa_z_prefixem = f"{folder_nazwa}_{nazwa_bez_rozszerzenia}"
                        
                        obraz = cv2.imread(pelna_sciezka)
                        if obraz is not None:
                            oryginalne_obrazy[nazwa_z_prefixem] = obraz
                            liczba_obrazow_folder[folder_nazwa] += 1
                        else:
                            print(f"Nie udało się wczytać obrazu: {pelna_sciezka}")
                    except Exception as e:
                        print(f"Błąd podczas wczytywania pliku {pelna_sciezka}: {str(e)}")
        except Exception as e:
            print(f"Błąd podczas przeglądania folderu {folder}: {str(e)}")
    
    # Wyświetlamy statystyki wczytanych obrazów
    print(f"Wczytano łącznie {len(oryginalne_obrazy)} oryginalnych obrazów")
    for folder_nazwa, liczba in liczba_obrazow_folder.items():
        print(f"  - Z folderu {folder_nazwa}: {liczba} obrazów")
    
    if not oryginalne_obrazy:
        print("Nie wczytano żadnych obrazów. Kończę działanie.")
        return
    
    # Inicjalizacja modeli OCR
    if args.metoda in ['easyocr', 'hybrid']:
        print("Inicjalizacja modelu EasyOCR...")
        easyocr_reader = easyocr.Reader(['pl', 'en'])
    
    # Przygotowujemy słowniki na wyniki
    obrazy_z_oznaczeniami = {}
    wszystkie_wyniki = {}
    wyniki_wg_folderow = {}
    
    # Przetwarzamy każdy obraz
    for nazwa, oryginalny_obraz in oryginalne_obrazy.items():
        print(f"Przetwarzanie obrazu: {nazwa}...")
        
        # Wykrywamy tekst odpowiednią metodą
        if args.metoda == 'easyocr':
            wykryte_teksty = wykryj_tekst_easyocr(oryginalny_obraz, easyocr_reader, args.jezyk, args.prog)
        elif args.metoda == 'tesseract':
            wykryte_teksty = wykryj_tekst_tesseract(oryginalny_obraz, args.jezyk)
        else:  # hybrid
            wykryte_easyocr = wykryj_tekst_easyocr(oryginalny_obraz, easyocr_reader, args.jezyk, args.prog)
            wykryte_tesseract = wykryj_tekst_tesseract(oryginalny_obraz, args.jezyk)
            wykryte_teksty = polacz_wyniki_ocr(wykryte_easyocr, wykryte_tesseract)
        
        # Zapisujemy wyniki
        wszystkie_wyniki[nazwa] = wykryte_teksty
        
        # Grupujemy wyniki według folderów
        folder_nazwa = nazwa.split('_')[0]
        if folder_nazwa not in wyniki_wg_folderow:
            wyniki_wg_folderow[folder_nazwa] = {}
        wyniki_wg_folderow[folder_nazwa][nazwa] = wykryte_teksty
        
        # Oznaczamy wykryty tekst na oryginalnym obrazie
        obraz_z_oznaczeniami = oznacz_wykryty_tekst(
            oryginalny_obraz, wykryte_teksty, args.prog
        )
        
        # Dodajemy do słownika wyników
        obrazy_z_oznaczeniami[nazwa] = obraz_z_oznaczeniami
    
    # Zapisujemy wyniki
    zapisz_wyniki(obrazy_z_oznaczeniami)
    
    # Czas zakończenia przetwarzania
    czas_koniec = time.time()
    print(f"Całkowity czas przetwarzania: {czas_koniec - czas_start:.2f} sekund")
    
    # Wyświetlamy podsumowanie dla wszystkich folderów
    for folder_nazwa, wyniki in wyniki_wg_folderow.items():
        liczba_wykrytych = sum(len(teksty) for teksty in wyniki.values())
        print(f"Folder {folder_nazwa}: wykryto {liczba_wykrytych} fragmentów tekstu na {len(wyniki)} obrazach.")
    
    # Wyświetlamy ogólne podsumowanie
    liczba_wykrytych_tekstow = sum(len(wyniki) for wyniki in wszystkie_wyniki.values())
    print(f"Wykryto łącznie {liczba_wykrytych_tekstow} fragmentów tekstu na {len(wszystkie_wyniki)} obrazach.")

if __name__ == "__main__":
    main()
