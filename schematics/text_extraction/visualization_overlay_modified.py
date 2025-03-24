import cv2
import numpy as np
import os
import json
from pathlib import Path
import argparse

# Definicja ścieżki bazowej projektu - bieżący katalog
PROJEKT_DIR = os.path.dirname(os.path.abspath(__file__))

# Wbudowane ścieżki względem projektu
FOLDER_COMBINED = os.path.join(PROJEKT_DIR, "combined_json")
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

def dostosuj_wspolrzedne(coords, image_size_json, image_size_actual, blok=None):
    """
    Dostosowuje współrzędne prostokąta z pliku JSON do rzeczywistego rozmiaru obrazu.
    
    Args:
        coords: Współrzędne [x_min, y_min, x_max, y_max]
        image_size_json: Rozmiar obrazu z pliku JSON (szerokość, wysokość)
        image_size_actual: Rzeczywisty rozmiar obrazu (szerokość, wysokość)
        blok: Opcjonalny blok zawierający metadane o współrzędnych
        
    Returns:
        Dostosowane współrzędne [x_min, y_min, x_max, y_max]
    """
    # Sprawdzamy czy blok zostały już przeskalowany
    if blok and blok.get("coords_scaled", False):
        # Jeśli tak, dostosowujemy tylko do rzeczywistego rozmiaru obrazu (bez ponownego skalowania)
        # ale wciąż zachowujemy oryginalną logikę dostosowania rozmiaru obrazu do rzeczywistego rozmiaru
        pass
    
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

def podziel_bloki_po_zrodle(bloki):
    """
    Dzieli bloki na grupy według źródła.
    
    Args:
        bloki: Lista bloków z połączonego pliku JSON
        
    Returns:
        Tuple (bloki_annotations, bloki_text_marked)
    """
    bloki_annotations = []
    bloki_text_marked = []
    
    for blok in bloki:
        if blok.get("source") == "annotations":
            bloki_annotations.append(blok)
        elif blok.get("source") == "text_marked":
            bloki_text_marked.append(blok)
    
    return bloki_annotations, bloki_text_marked

def znajdz_zachodzace_obszary(bloki_annotations, bloki_text_marked, prog_iou=0.5):
    """
    Znajduje obszary, które zachodzą na siebie w obu zestawach bloków.
    
    Args:
        bloki_annotations: Lista bloków z source="annotations"
        bloki_text_marked: Lista bloków z source="text_marked"
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

def narysuj_oznaczenia(sciezka_obrazu, dane_combined, folder_wyjsciowy, tryb='standard'):
    """
    Rysuje oznaczenia na obrazie i zapisuje wynik do folderu wyjściowego.
    
    Args:
        sciezka_obrazu: Ścieżka do pliku obrazu
        dane_combined: Dane z połączonego pliku JSON
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
        
        # Pobieramy rozmiar z pliku JSON
        image_size_json = None
        if "image_size" in dane_combined:
            image_size_json = (dane_combined["image_size"]["width"], 
                              dane_combined["image_size"]["height"])
        
        # Sprawdzamy, czy mamy informacje o różnych rozmiarach obrazów
        niezgodne_rozmiary = False
        wymaga_skalowania = False
        
        if "image_sizes" in dane_combined:
            # Sprawdź flagę, czy wymaga skalowania
            wymaga_skalowania = dane_combined["image_sizes"].get("wymaga_skalowania", False)
            
            # Dla zachowania kompatybilności wstecz, sprawdź też flagę zgodne_rozmiary
            if not dane_combined["image_sizes"].get("zgodne_rozmiary", True):
                niezgodne_rozmiary = True
            
            # Wyświetl informacje o skalowaniu, jeśli jest potrzebne
            if wymaga_skalowania:
                print(f"Obraz {os.path.basename(sciezka_obrazu)} wymaga skalowania")
                
                if dane_combined["image_sizes"].get("annotations") and dane_combined["image_sizes"].get("text_marked"):
                    print(f"  Annotations: {dane_combined['image_sizes']['annotations']}")
                    print(f"  Text_marked: {dane_combined['image_sizes']['text_marked']}")
                    print(f"  Współczynniki skalowania: x={dane_combined['image_sizes']['skala_x']:.2f}, y={dane_combined['image_sizes']['skala_y']:.2f}")
        
        print(f"Obraz: {os.path.basename(sciezka_obrazu)}")
        print(f"  Rozmiar rzeczywisty: {actual_size}")
        print(f"  Rozmiar z JSON: {image_size_json}")
        
        # Pobierz wszystkie bloki
        wszystkie_bloki = dane_combined.get("blocks", [])
        
        if tryb == 'standard':
            # Rysujemy oznaczenia z różnymi kolorami w zależności od źródła
            for blok in wszystkie_bloki:
                if "coords" in blok:
                    # Pobieramy współrzędne prostokąta i dostosowujemy je
                    coords = blok["coords"]
                    x_min, y_min, x_max, y_max = dostosuj_wspolrzedne(coords, image_size_json, actual_size, blok)
                    
                    # Wybieramy kolor w zależności od źródła
                    if blok.get("source") == "annotations":
                        kolor = (255, 0, 0)  # Niebieski (BGR) dla annotations
                    else:
                        kolor = (0, 255, 0)  # Zielony dla text_marked
                    
                    # Rysujemy prostokąt
                    cv2.rectangle(obraz_z_oznaczeniami, (x_min, y_min), (x_max, y_max), kolor, 2)
                    
                    # Jeśli blok został przeskalowany, dodaj oznaczenie
                    if (wymaga_skalowania or niezgodne_rozmiary) and blok.get("coords_scaled"):
                        cv2.putText(
                            obraz_z_oznaczeniami, 
                            "S", 
                            (x_max - 15, y_min + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            kolor, 
                            2
                        )
        
        elif tryb == 'roznice':
            # Dzielimy bloki według źródeł
            bloki_annotations, bloki_text_marked = podziel_bloki_po_zrodle(wszystkie_bloki)
            
            # Dostosowujemy współrzędne wszystkich bloków
            for blok in wszystkie_bloki:
                if "coords" in blok:
                    blok["coords"] = dostosuj_wspolrzedne(blok["coords"], image_size_json, actual_size, blok)
            
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
                
                # Jeśli blok został przeskalowany, dodaj oznaczenie
                if (wymaga_skalowania or niezgodne_rozmiary) and (blok_ann.get("coords_scaled") or blok_text.get("coords_scaled")):
                    cv2.putText(
                        obraz_z_oznaczeniami, 
                        "S", 
                        (x_max_ann - 15, y_min_ann + 15), 
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
            
            # Na koniec rysujemy pozostałe prostokąty z text_marked (na zielono)
            for blok in bloki_text_marked:
                if id(blok) not in narysowane_text_ids and "coords" in blok:
                    bbox = blok["coords"]
                    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
                    cv2.rectangle(obraz_z_oznaczeniami, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Jeśli blok został przeskalowany, dodaj oznaczenie
                    if (wymaga_skalowania or niezgodne_rozmiary) and blok.get("coords_scaled"):
                        cv2.putText(
                            obraz_z_oznaczeniami, 
                            "S", 
                            (x_max - 15, y_min + 15), 
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
    parser = argparse.ArgumentParser(description='Wizualizacja oznaczeń z połączonych plików JSON')
    parser.add_argument('--combined', type=str, default=FOLDER_COMBINED,
                        help=f'Ścieżka do folderu z połączonymi plikami JSON (domyślnie: {FOLDER_COMBINED})')
    parser.add_argument('--output', type=str, default=FOLDER_OUTPUT,
                        help=f'Ścieżka do folderu wyjściowego z obrazami (domyślnie: {FOLDER_OUTPUT})')
    parser.add_argument('--images', type=str, default=FOLDER_IMAGES,
                        help=f'Ścieżka do folderu z obrazami (domyślnie: {FOLDER_IMAGES})')
    parser.add_argument('--tryb', type=str, choices=['standard', 'roznice'], default='standard',
                        help='Tryb wizualizacji: standard (nakładanie oznaczeń) lub roznice (wyróżnianie różnic)')
    args = parser.parse_args()
    
    # Wczytaj pliki JSON z folderu połączonych plików
    pliki_combined = wczytaj_pliki_json(args.combined)
    
    # Jeśli podano folder z obrazami, wyszukaj w nim pliki
    pliki_obrazow = {}
    if os.path.isdir(args.images):
        pliki_obrazow = znajdz_pliki_obrazow(args.images)
    
    # Licznik pomyślnie przetworzonych obrazów
    liczba_przetworzonych = 0
    
    # Przetwórz każdy plik JSON
    for nazwa_pliku, dane_combined in pliki_combined.items():
        # Pobierz ścieżkę do obrazu z pliku JSON
        sciezka_obrazu_z_jsona = dane_combined.get("image_path", "")
        
        # Normalizacja ścieżki - bierzemy tylko nazwę pliku
        nazwa_obrazu = os.path.basename(sciezka_obrazu_z_jsona) if sciezka_obrazu_z_jsona else f"{nazwa_pliku}.png"
        
        # Szukamy pełnej ścieżki do obrazu
        pelna_sciezka = None
        
        # 1. Najpierw sprawdź, czy plik istnieje bezpośrednio
        if os.path.exists(nazwa_obrazu):
            pelna_sciezka = nazwa_obrazu
        
        # 2. Sprawdź w słowniku znalezionych plików obrazów
        elif nazwa_obrazu in pliki_obrazow:
            pelna_sciezka = pliki_obrazow[nazwa_obrazu]
        
        # 3. Spróbuj znaleźć obraz w znanych folderach projektu
        else:
            pelna_sciezka = znajdz_obrazy_w_folderach(args.images, nazwa_obrazu)
            
            # Jeśli nadal nie znaleziono, sprawdź ścieżkę z pliku JSON
            if not pelna_sciezka and sciezka_obrazu_z_jsona:
                # Sprawdź pełną ścieżkę
                if os.path.exists(sciezka_obrazu_z_jsona):
                    pelna_sciezka = sciezka_obrazu_z_jsona
                
                # Sprawdź ścieżkę względem katalogu projektu
                sciezka_wzgledna = os.path.join(PROJEKT_DIR, sciezka_obrazu_z_jsona)
                if os.path.exists(sciezka_wzgledna):
                    pelna_sciezka = sciezka_wzgledna
        
        if pelna_sciezka:
            # Narysuj oznaczenia na obrazie i zapisz wynik
            if narysuj_oznaczenia(pelna_sciezka, dane_combined, args.output, args.tryb):
                liczba_przetworzonych += 1
                print(f"Pomyślnie przetworzono: {nazwa_pliku} -> {os.path.basename(pelna_sciezka)}")
        else:
            print(f"Nie znaleziono obrazu dla pliku: {nazwa_pliku}")
            print(f"  Ścieżka w JSON: {sciezka_obrazu_z_jsona}")
    
    print(f"\nPomyślnie przetworzono {liczba_przetworzonych} obrazów z {len(pliki_combined)} plików JSON")
    print(f"Wyniki zapisano w folderze: {args.output}")

if __name__ == "__main__":
    main() 