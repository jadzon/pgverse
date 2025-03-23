import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse

# Definicja ścieżki bazowej projektu - bieżący katalog
PROJEKT_DIR = os.path.dirname(os.path.abspath(__file__))

# Wbudowane ścieżki względem projektu
FOLDER_ANNOTATIONS = os.path.join(PROJEKT_DIR, "annotations")
FOLDER_TEXT_MARKED = os.path.join(PROJEKT_DIR, "text_marked")
FOLDER_OUTPUT = os.path.join(PROJEKT_DIR, "visualization_output")
FOLDER_IMAGES = os.path.join(PROJEKT_DIR, "Dataset")

def wczytaj_pliki_json(folder):
    """
    Wczytuje wszystkie pliki JSON z podanego folderu.
    
    Args:
        folder: Ścieżka do folderu z plikami JSON
        
    Returns:
        Słownik {nazwa_pliku: dane_json}
    """
    pliki_json = {}
    sciezka_folderu = Path(folder)
    
    if not sciezka_folderu.exists():
        print(f"Folder {folder} nie istnieje!")
        return pliki_json
    
    for plik in sciezka_folderu.glob("**/*.json"):
        try:
            with open(plik, 'r', encoding='utf-8') as f:
                dane = json.load(f)
                nazwa_pliku = plik.stem
                pliki_json[nazwa_pliku] = dane
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku {plik}: {e}")
    
    print(f"Wczytano {len(pliki_json)} plików JSON z folderu {folder}")
    return pliki_json

def znajdz_pary_plikow_po_obrazie(pliki_annotations, pliki_text_marked):
    """
    Znajduje pary plików na podstawie ścieżki do obrazu w plikach JSON.
    
    Args:
        pliki_annotations: Słownik plików z folderu annotations
        pliki_text_marked: Słownik plików z folderu text_marked
        
    Returns:
        Słownik {sciezka_obrazu: (dane_annotations, dane_text_marked)}
    """
    pary = {}
    
    # Tworzenie mapy ścieżek obrazów do danych JSON z annotations
    obrazy_annotations = {}
    for nazwa, dane in pliki_annotations.items():
        if "image_path" in dane:
            sciezka = dane["image_path"]
            # Normalizacja ścieżki - bierzemy tylko nazwę pliku
            sciezka = os.path.basename(sciezka)
            obrazy_annotations[sciezka] = dane
    
    # Znajdowanie odpowiadających sobie plików z text_marked
    for nazwa, dane in pliki_text_marked.items():
        if "image_path" in dane:
            sciezka = dane["image_path"]
            # Normalizacja ścieżki - bierzemy tylko nazwę pliku
            sciezka = os.path.basename(sciezka)
            if sciezka in obrazy_annotations:
                pary[sciezka] = (obrazy_annotations[sciezka], dane)
    
    print(f"Znaleziono {len(pary)} par plików na podstawie ścieżek obrazów")
    return pary

def znajdz_pliki_obrazow(folder_obrazow):
    """
    Wyszukuje wszystkie pliki obrazów w podanym folderze i podfolderach.
    
    Args:
        folder_obrazow: Ścieżka do folderu z obrazami
        
    Returns:
        Słownik {nazwa_pliku: pełna_ścieżka}
    """
    pliki_obrazow = {}
    
    for rozszerzenie in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        for sciezka in Path(folder_obrazow).rglob(f"*{rozszerzenie}"):
            nazwa_pliku = sciezka.name
            pliki_obrazow[nazwa_pliku] = str(sciezka)
    
    print(f"Znaleziono {len(pliki_obrazow)} plików obrazów w folderze {folder_obrazow}")
    return pliki_obrazow

def polacz_oznaczenia(dane_annotations, dane_text_marked):
    """
    Łączy oznaczenia z obu plików JSON, dodając informację o źródle.
    
    Args:
        dane_annotations: Dane z pliku JSON z folderu annotations
        dane_text_marked: Dane z pliku JSON z folderu text_marked
        
    Returns:
        Lista wszystkich bloków z obu plików
    """
    bloki_polaczone = []
    
    # Dodajemy bloki z annotations (jeśli istnieją)
    if "blocks" in dane_annotations:
        for blok in dane_annotations["blocks"]:
            # Dodajemy źródło bloku
            blok["source"] = "annotations"
            bloki_polaczone.append(blok)
    
    # Dodajemy bloki z text_marked (jeśli istnieją)
    if "blocks" in dane_text_marked:
        for blok in dane_text_marked["blocks"]:
            # Dodajemy źródło bloku
            blok["source"] = "text_marked"
            bloki_polaczone.append(blok)
    
    return bloki_polaczone

def oblicz_iou(bbox1, bbox2):
    """
    Oblicza IoU (Intersection over Union) dla dwóch bounding boxów.
    
    Args:
        bbox1: Pierwszy bounding box [x_min, y_min, x_max, y_max]
        bbox2: Drugi bounding box [x_min, y_min, x_max, y_max]
        
    Returns:
        Wartość IoU (0.0 - 1.0)
    """
    # Rozpakowanie koordynatów
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Obliczenie współrzędnych przecięcia
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    # Sprawdzenie, czy prostokąty się przecinają
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Obliczenie pola przecięcia
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Obliczenie pól obu prostokątów
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Obliczenie pola sumy (suma - przecięcie)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Obliczenie IoU
    iou = intersection_area / float(union_area)
    
    return iou

def znajdz_zachodzace_obszary(bloki_annotations, bloki_text_marked, prog_iou=0.5):
    """
    Znajduje obszary, które zachodzą na siebie w obu zestawach bloków.
    
    Args:
        bloki_annotations: Lista bloków z pliku annotations
        bloki_text_marked: Lista bloków z pliku text_marked
        prog_iou: Próg IoU dla uznania dwóch bboxów za nakładające się
        
    Returns:
        Lista par nakładających się bloków (blok_annotations, blok_text_marked, iou)
    """
    zachodzace_obszary = []
    
    for blok_ann in bloki_annotations:
        if "coords" not in blok_ann:
            continue
        
        bbox_ann = blok_ann["coords"]
        
        for blok_text in bloki_text_marked:
            if "coords" not in blok_text:
                continue
            
            bbox_text = blok_text["coords"]
            
            iou = oblicz_iou(bbox_ann, bbox_text)
            if iou >= prog_iou:
                zachodzace_obszary.append((blok_ann, blok_text, iou))
    
    return zachodzace_obszary

def dostosuj_wspolrzedne(coords, image_size_json, image_size_actual):
    """
    Dostosowuje współrzędne prostokąta z pliku JSON do rzeczywistego rozmiaru obrazu.
    
    Args:
        coords: Współrzędne [x_min, y_min, x_max, y_max]
        image_size_json: Rozmiar obrazu z pliku JSON (szerokość, wysokość)
        image_size_actual: Rzeczywisty rozmiar obrazu (szerokość, wysokość)
        
    Returns:
        Dostosowane współrzędne [x_min, y_min, x_max, y_max]
    """
    # Sprawdzamy typ danych współrzędnych
    if not isinstance(coords, list) or len(coords) != 4:
        print(f"Błędny format współrzędnych: {coords}")
        return coords
    
    try:
        # Rozpakowanie współrzędnych
        x_min, y_min, x_max, y_max = [float(coord) for coord in coords]
        
        # Jeśli rozmiar z JSON jest dostępny i różni się od rzeczywistego, dostosuj współrzędne
        if image_size_json and image_size_actual and (image_size_json != image_size_actual):
            json_width, json_height = image_size_json
            actual_width, actual_height = image_size_actual
            
            # Obliczanie współczynników skalowania
            scale_x = actual_width / json_width
            scale_y = actual_height / json_height
            
            # Skalowanie współrzędnych
            x_min = int(x_min * scale_x)
            y_min = int(y_min * scale_y)
            x_max = int(x_max * scale_x)
            y_max = int(y_max * scale_y)
        
        return [int(x_min), int(y_min), int(x_max), int(y_max)]
    except Exception as e:
        print(f"Błąd podczas dostosowywania współrzędnych: {e}, coords: {coords}")
        return coords

def narysuj_oznaczenia(sciezka_obrazu, dane_annotations, dane_text_marked, folder_wyjsciowy, tryb='standard'):
    """
    Rysuje oznaczenia na obrazie i zapisuje wynik do folderu wyjściowego.
    
    Args:
        sciezka_obrazu: Ścieżka do pliku obrazu
        dane_annotations: Dane z pliku JSON z folderu annotations
        dane_text_marked: Dane z pliku JSON z folderu text_marked
        folder_wyjsciowy: Folder, w którym zostanie zapisany obraz z oznaczeniami
        tryb: Tryb rysowania ('standard' lub 'roznice')
        
    Returns:
        True jeśli operacja się powiodła, False w przeciwnym przypadku
    """
    try:
        obraz = cv2.imread(sciezka_obrazu)
        if obraz is None:
            print(f"Nie udało się wczytać obrazu: {sciezka_obrazu}")
            return False
        
        # Kopiujemy obraz, aby nie modyfikować oryginału
        obraz_z_oznaczeniami = obraz.copy()
        
        # Pobieramy rzeczywisty rozmiar obrazu
        actual_height, actual_width = obraz.shape[:2]
        actual_size = (actual_width, actual_height)
        
        # Pobieramy rozmiary z plików JSON
        image_size_annotations = None
        if "image_size" in dane_annotations:
            image_size_annotations = (dane_annotations["image_size"]["width"], 
                                     dane_annotations["image_size"]["height"])
        
        image_size_text_marked = None
        if "image_size" in dane_text_marked:
            image_size_text_marked = (dane_text_marked["image_size"]["width"], 
                                     dane_text_marked["image_size"]["height"])
        
        print(f"Obraz: {os.path.basename(sciezka_obrazu)}")
        print(f"  Rozmiar rzeczywisty: {actual_size}")
        print(f"  Rozmiar z annotations: {image_size_annotations}")
        print(f"  Rozmiar z text_marked: {image_size_text_marked}")
        
        if tryb == 'standard':
            # Łączymy wszystkie oznaczenia
            bloki_polaczone = polacz_oznaczenia(dane_annotations, dane_text_marked)
            
            # Rysujemy oznaczenia z różnymi kolorami w zależności od źródła
            for blok in bloki_polaczone:
                if "coords" in blok:
                    # Wybieramy odpowiedni rozmiar obrazu do skalowania
                    image_size_json = None
                    if blok.get("source") == "annotations" and image_size_annotations:
                        image_size_json = image_size_annotations
                    elif blok.get("source") == "text_marked" and image_size_text_marked:
                        image_size_json = image_size_text_marked
                    
                    # Pobieramy współrzędne prostokąta i dostosowujemy je
                    coords = blok["coords"]
                    x_min, y_min, x_max, y_max = dostosuj_wspolrzedne(coords, image_size_json, actual_size)
                    
                    # Wybieramy kolor w zależności od źródła
                    if blok.get("source") == "annotations":
                        kolor = (255, 0, 0)  # Niebieski (BGR) dla annotations
                    else:
                        kolor = (0, 255, 0)  # Zielony dla text_marked
                    
                    # Rysujemy prostokąt
                    cv2.rectangle(obraz_z_oznaczeniami, (x_min, y_min), (x_max, y_max), kolor, 2)
                    
                    # Dodajemy tekst, jeśli jest dostępny
                    if "text" in blok:
                        tekst = blok["text"]
                        # Skracamy tekst, jeśli jest za długi
                        if len(tekst) > 20:
                            tekst = tekst[:17] + "..."
                        
                        # Dodajemy tło pod tekstem
                        (text_width, text_height), _ = cv2.getTextSize(
                            tekst, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        cv2.rectangle(
                            obraz_z_oznaczeniami,
                            (x_min, y_min - text_height - 10),
                            (x_min + text_width, y_min),
                            (0, 0, 0),
                            -1
                        )
                        
                        # Dodajemy tekst
                        cv2.putText(
                            obraz_z_oznaczeniami, 
                            tekst, 
                            (x_min, y_min - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 255, 255), 
                            2
                        )
        
        elif tryb == 'roznice':
            # Pobieramy bloki z obu plików
            bloki_annotations = dane_annotations.get("blocks", [])
            bloki_text_marked = dane_text_marked.get("blocks", [])
            
            # Dostosowujemy współrzędne bloków
            for blok in bloki_annotations:
                if "coords" in blok:
                    blok["coords"] = dostosuj_wspolrzedne(blok["coords"], 
                                                         image_size_annotations, 
                                                         actual_size)
            
            for blok in bloki_text_marked:
                if "coords" in blok:
                    blok["coords"] = dostosuj_wspolrzedne(blok["coords"], 
                                                         image_size_text_marked, 
                                                         actual_size)
            
            # Znajdujemy nakładające się obszary
            zachodzace_obszary = znajdz_zachodzace_obszary(bloki_annotations, bloki_text_marked, 0.5)
            
            # Tworzymy zbiory identyfikatorów bloków, które już zostały narysowane jako nakładające się
            narysowane_ann_ids = set()
            narysowane_text_ids = set()
            
            # Najpierw rysujemy nakładające się obszary
            for i, (blok_ann, blok_text, iou) in enumerate(zachodzace_obszary):
                bbox_ann = blok_ann["coords"]
                bbox_text = blok_text["coords"]
                
                # Dodajemy identyfikatory do zbiorów narysowanych
                narysowane_ann_ids.add(id(blok_ann))
                narysowane_text_ids.add(id(blok_text))
                
                # Rysujemy oba prostokąty na żółto (obszary nakładające się)
                x_min_ann, y_min_ann, x_max_ann, y_max_ann = [int(coord) for coord in bbox_ann]
                cv2.rectangle(obraz_z_oznaczeniami, (x_min_ann, y_min_ann), (x_max_ann, y_max_ann), (0, 255, 255), 2)
                
                # Dodajemy informację o IoU
                tekst = f"IoU: {iou:.2f}"
                cv2.putText(
                    obraz_z_oznaczeniami, 
                    tekst, 
                    (x_min_ann, y_min_ann - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 255), 
                    2
                )
            
            # Następnie rysujemy pozostałe prostokąty z annotations (na niebiesko)
            for blok in bloki_annotations:
                if id(blok) not in narysowane_ann_ids and "coords" in blok:
                    bbox = blok["coords"]
                    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
                    cv2.rectangle(obraz_z_oznaczeniami, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    
                    if "text" in blok:
                        tekst = blok["text"]
                        if len(tekst) > 20:
                            tekst = tekst[:17] + "..."
                        
                        cv2.putText(
                            obraz_z_oznaczeniami, 
                            tekst, 
                            (x_min, y_min - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 0, 0), 
                            2
                        )
            
            # Na koniec rysujemy pozostałe prostokąty z text_marked (na zielono)
            for blok in bloki_text_marked:
                if id(blok) not in narysowane_text_ids and "coords" in blok:
                    bbox = blok["coords"]
                    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
                    cv2.rectangle(obraz_z_oznaczeniami, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    if "text" in blok:
                        tekst = blok["text"]
                        if len(tekst) > 20:
                            tekst = tekst[:17] + "..."
                        
                        cv2.putText(
                            obraz_z_oznaczeniami, 
                            tekst, 
                            (x_min, y_min - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            2
                        )
        
        # Tworzymy folder wyjściowy, jeśli nie istnieje
        os.makedirs(folder_wyjsciowy, exist_ok=True)
        
        # Zapisujemy obraz z oznaczeniami
        nazwa_pliku = os.path.basename(sciezka_obrazu)
        if tryb == 'roznice':
            nazwa_pliku = f"diff_{nazwa_pliku}"
        sciezka_wyjsciowa = os.path.join(folder_wyjsciowy, nazwa_pliku)
        cv2.imwrite(sciezka_wyjsciowa, obraz_z_oznaczeniami)
        
        return True
    
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu {sciezka_obrazu}: {e}")
        return False

def znajdz_obrazy_w_folderach(foldery_obrazow, nazwa_pliku):
    """
    Wyszukuje plik obrazu w podanych folderach.
    
    Args:
        foldery_obrazow: Lista folderów do przeszukania
        nazwa_pliku: Nazwa szukanego pliku
        
    Returns:
        Pełna ścieżka do znalezionego pliku lub None
    """
    # Lista możliwych podfolderów projektu, które mogą zawierać obrazy
    podkatalogu_do_sprawdzenia = [
        "", # główny katalog
        "Dataset",
        "Dataset/Elektroniczne",
        "Dataset/Automatyka",
        "Dataset/Przetworzone"
    ]
    
    # Dodaj foldery przekazane jako argument
    if isinstance(foldery_obrazow, str) and foldery_obrazow:
        podkatalogu_do_sprawdzenia.append(foldery_obrazow)
    elif isinstance(foldery_obrazow, list):
        podkatalogu_do_sprawdzenia.extend(foldery_obrazow)
    
    # Sprawdź wszystkie możliwe lokalizacje
    for folder in podkatalogu_do_sprawdzenia:
        pelna_sciezka = os.path.join(PROJEKT_DIR, folder, nazwa_pliku)
        if os.path.exists(pelna_sciezka):
            return pelna_sciezka
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Łączenie oznaczeń z plików JSON i nakładanie ich na obrazy')
    parser.add_argument('--annotations', type=str, default=FOLDER_ANNOTATIONS,
                        help=f'Ścieżka do folderu z plikami JSON z annotations (domyślnie: {FOLDER_ANNOTATIONS})')
    parser.add_argument('--text_marked', type=str, default=FOLDER_TEXT_MARKED,
                        help=f'Ścieżka do folderu z plikami JSON z text_marked (domyślnie: {FOLDER_TEXT_MARKED})')
    parser.add_argument('--output', type=str, default=FOLDER_OUTPUT,
                        help=f'Ścieżka do folderu wyjściowego z obrazami (domyślnie: {FOLDER_OUTPUT})')
    parser.add_argument('--images', type=str, default=FOLDER_IMAGES,
                        help=f'Ścieżka do folderu z obrazami (domyślnie: {FOLDER_IMAGES})')
    parser.add_argument('--tryb', type=str, choices=['standard', 'roznice'], default='standard',
                        help='Tryb wizualizacji: standard (nakładanie oznaczeń) lub roznice (wyróżnianie różnic)')
    args = parser.parse_args()
    
    # Wczytaj pliki JSON z obu folderów
    pliki_annotations = wczytaj_pliki_json(args.annotations)
    pliki_text_marked = wczytaj_pliki_json(args.text_marked)
    
    # Znajdź pary plików na podstawie ścieżek obrazów
    pary_plikow = znajdz_pary_plikow_po_obrazie(pliki_annotations, pliki_text_marked)
    
    # Jeśli podano folder z obrazami, wyszukaj w nim pliki
    pliki_obrazow = {}
    if os.path.isdir(args.images):
        pliki_obrazow = znajdz_pliki_obrazow(args.images)
    
    # Licznik pomyślnie przetworzonych obrazów
    liczba_przetworzonych = 0
    
    # Przetwórz każdą parę plików
    for sciezka_obrazu, (dane_annotations, dane_text_marked) in pary_plikow.items():
        # Sprawdź, czy ścieżka jest względna i czy plik istnieje
        pelna_sciezka = None
        
        # 1. Najpierw sprawdź, czy plik istnieje bezpośrednio
        if os.path.exists(sciezka_obrazu):
            pelna_sciezka = sciezka_obrazu
        
        # 2. Sprawdź w słowniku znalezionych plików obrazów
        elif sciezka_obrazu in pliki_obrazow:
            pelna_sciezka = pliki_obrazow[sciezka_obrazu]
        
        # 3. Spróbuj znaleźć obraz w znanych folderach projektu
        else:
            pelna_sciezka = znajdz_obrazy_w_folderach(args.images, sciezka_obrazu)
            
            # Jeśli nadal nie znaleziono, sprawdź ścieżki z plików JSON
            if not pelna_sciezka:
                for potencjalna_sciezka in [
                    dane_annotations.get("image_path", ""),
                    dane_text_marked.get("image_path", "")
                ]:
                    if potencjalna_sciezka:
                        # Sprawdź pełną ścieżkę
                        if os.path.exists(potencjalna_sciezka):
                            pelna_sciezka = potencjalna_sciezka
                            break
                        
                        # Sprawdź ścieżkę względem katalogu projektu
                        sciezka_wzgledna = os.path.join(PROJEKT_DIR, potencjalna_sciezka)
                        if os.path.exists(sciezka_wzgledna):
                            pelna_sciezka = sciezka_wzgledna
                            break
                        
                        # Sprawdź samą nazwę pliku w folderach projektu
                        nazwa_pliku = os.path.basename(potencjalna_sciezka)
                        znaleziona_sciezka = znajdz_obrazy_w_folderach(args.images, nazwa_pliku)
                        if znaleziona_sciezka:
                            pelna_sciezka = znaleziona_sciezka
                            break
        
        if pelna_sciezka:
            # Narysuj oznaczenia na obrazie i zapisz wynik
            if narysuj_oznaczenia(pelna_sciezka, dane_annotations, dane_text_marked, args.output, args.tryb):
                liczba_przetworzonych += 1
        else:
            # Wypisz ścieżki z JSON-ów dla debugowania
            print(f"Nie znaleziono obrazu: {sciezka_obrazu}")
            print(f"  Ścieżka w annotations: {dane_annotations.get('image_path', 'brak')}")
            print(f"  Ścieżka w text_marked: {dane_text_marked.get('image_path', 'brak')}")
    
    print(f"Pomyślnie przetworzono {liczba_przetworzonych} obrazów z {len(pary_plikow)} par plików")
    print(f"Wyniki zapisano w folderze: {args.output}")

if __name__ == "__main__":
    main() 