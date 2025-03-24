import os
import json
import argparse
from pathlib import Path

# Definicja ścieżki bazowej projektu - bieżący katalog
PROJEKT_DIR = os.path.dirname(os.path.abspath(__file__))

# Wbudowane ścieżki względem projektu
FOLDER_ANNOTATIONS = os.path.join(PROJEKT_DIR, "annotations")
FOLDER_TEXT_MARKED = os.path.join(PROJEKT_DIR, "text_marked")
FOLDER_COMBINED = os.path.join(PROJEKT_DIR, "combined_json")

def wczytaj_pliki_json(folder):
    """
    Wczytuje wszystkie pliki JSON z podanego folderu i podfolderów.
    
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
                # Użyj ścieżki względnej jako klucza
                sciezka_wzgledna = os.path.relpath(plik, folder)
                pliki_json[sciezka_wzgledna] = dane
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku {plik}: {e}")
    
    print(f"Wczytano {len(pliki_json)} plików JSON z folderu {folder}")
    return pliki_json

def porownaj_rozmiary_obrazow(dane_annotations, dane_text_marked):
    """
    Porównuje rozmiary obrazów z dwóch plików JSON i zawsze zwraca informacje o skalowaniu.
    
    Args:
        dane_annotations: Dane z pliku JSON z annotations
        dane_text_marked: Dane z pliku JSON z text_marked
        
    Returns:
        (dict) - Informacje o rozmiarach obrazów i skalowaniu
    """
    info_rozmiary = {
        "annotations_size": None,
        "text_marked_size": None,
        "zgodne_rozmiary": True,
        "skala_x": 1.0,
        "skala_y": 1.0,
        "wymaga_skalowania": False
    }
    
    # Sprawdź czy pliki mają informacje o rozmiarze obrazu
    if "image_size" in dane_annotations:
        info_rozmiary["annotations_size"] = (
            dane_annotations["image_size"].get("width", 0), 
            dane_annotations["image_size"].get("height", 0)
        )
    
    if "image_size" in dane_text_marked:
        info_rozmiary["text_marked_size"] = (
            dane_text_marked["image_size"].get("width", 0), 
            dane_text_marked["image_size"].get("height", 0)
        )
    
    # Jeśli oba pliki mają rozmiary, sprawdź zgodność i oblicz skalowanie
    if info_rozmiary["annotations_size"] and info_rozmiary["text_marked_size"]:
        ann_width, ann_height = info_rozmiary["annotations_size"]
        text_width, text_height = info_rozmiary["text_marked_size"]
        
        # Oblicz różnice procentowe tylko dla logowania
        if ann_width > 0 and ann_height > 0 and text_width > 0 and text_height > 0:
            roznica_x = abs(ann_width - text_width) / max(ann_width, text_width)
            roznica_y = abs(ann_height - text_height) / max(ann_height, text_height)
            
            # Ustaw flagę zgodności tylko dla celów informacyjnych
            info_rozmiary["zgodne_rozmiary"] = (roznica_x <= 0.05 and roznica_y <= 0.05)
            
            # Oblicz współczynniki skalowania (text_marked -> annotations)
            if ann_width > 0 and text_width > 0:
                info_rozmiary["skala_x"] = ann_width / text_width
            
            if ann_height > 0 and text_height > 0:
                info_rozmiary["skala_y"] = ann_height / text_height
            
            # Ustal czy wymaga skalowania - zawsze skaluj gdy rozmiary się różnią
            if ann_width != text_width or ann_height != text_height:
                info_rozmiary["wymaga_skalowania"] = True
    
    return info_rozmiary

def polacz_pliki_json(dane_annotations, dane_text_marked, info_rozmiary):
    """
    Łączy dane z dwóch plików JSON, zachowując tylko współrzędne prostokątów.
    
    Args:
        dane_annotations: Dane z pliku JSON z annotations
        dane_text_marked: Dane z pliku JSON z text_marked
        info_rozmiary: Informacje o rozmiarach obrazów
        
    Returns:
        Słownik z połączonymi danymi
    """
    polaczone_dane = {}
    
    # Sprawdź ścieżkę do obrazu (priorytet dla annotations)
    if "image_path" in dane_annotations:
        polaczone_dane["image_path"] = dane_annotations["image_path"]
    elif "image_path" in dane_text_marked:
        polaczone_dane["image_path"] = dane_text_marked["image_path"]
    
    # Dodaj informacje o rozmiarach obrazów
    polaczone_dane["image_sizes"] = {
        "annotations": info_rozmiary["annotations_size"],
        "text_marked": info_rozmiary["text_marked_size"],
        "zgodne_rozmiary": info_rozmiary["zgodne_rozmiary"],
        "skala_x": info_rozmiary["skala_x"],
        "skala_y": info_rozmiary["skala_y"],
        "wymaga_skalowania": info_rozmiary["wymaga_skalowania"]
    }
    
    # Ustaw rozmiar obrazu (priorytet dla annotations)
    if info_rozmiary["annotations_size"]:
        width, height = info_rozmiary["annotations_size"]
        polaczone_dane["image_size"] = {"width": width, "height": height}
    elif info_rozmiary["text_marked_size"]:
        width, height = info_rozmiary["text_marked_size"]
        polaczone_dane["image_size"] = {"width": width, "height": height}
    
    # Połącz bloki, oznaczając źródło
    polaczone_dane["blocks"] = []
    
    # Dodaj bloki z annotations - tylko współrzędne i źródło
    if "blocks" in dane_annotations and isinstance(dane_annotations["blocks"], list):
        for blok in dane_annotations["blocks"]:
            if "coords" in blok:
                blok_uproszczony = {
                    "coords": blok["coords"],
                    "source": "annotations"
                }
                
                # Dodaj typ jeśli istnieje
                if "type" in blok:
                    blok_uproszczony["type"] = blok["type"]
                
                polaczone_dane["blocks"].append(blok_uproszczony)
    
    # Dodaj bloki z text_marked - tylko współrzędne i źródło
    if "blocks" in dane_text_marked and isinstance(dane_text_marked["blocks"], list):
        for blok in dane_text_marked["blocks"]:
            if "coords" in blok:
                blok_uproszczony = {
                    "coords": blok["coords"],
                    "source": "text_marked"
                }
                
                # Dodaj typ jeśli istnieje
                if "type" in blok:
                    blok_uproszczony["type"] = blok["type"]
                
                # Zawsze skaluj współrzędne bloków z text_marked jeśli wymaga skalowania
                if info_rozmiary["wymaga_skalowania"]:
                    try:
                        coords = blok_uproszczony["coords"]
                        if isinstance(coords, list) and len(coords) == 4:
                            x_min, y_min, x_max, y_max = [float(coord) for coord in coords]
                            
                            # Dostosuj współrzędne do rozmiaru annotations
                            x_min = int(x_min * info_rozmiary["skala_x"])
                            y_min = int(y_min * info_rozmiary["skala_y"])
                            x_max = int(x_max * info_rozmiary["skala_x"])
                            y_max = int(y_max * info_rozmiary["skala_y"])
                            
                            blok_uproszczony["coords"] = [x_min, y_min, x_max, y_max]
                            blok_uproszczony["coords_scaled"] = True
                    except Exception as e:
                        print(f"Błąd podczas skalowania współrzędnych: {e}")
                
                polaczone_dane["blocks"].append(blok_uproszczony)
    
    return polaczone_dane

def zapisz_polaczony_plik(polaczone_dane, sciezka_wyjsciowa):
    """
    Zapisuje połączone dane do pliku JSON.
    
    Args:
        polaczone_dane: Dane do zapisania
        sciezka_wyjsciowa: Ścieżka do pliku wyjściowego
        
    Returns:
        True jeśli udało się zapisać plik, False w przeciwnym razie
    """
    try:
        # Utwórz katalog docelowy, jeśli nie istnieje
        os.makedirs(os.path.dirname(sciezka_wyjsciowa), exist_ok=True)
        
        # Zapisz plik
        with open(sciezka_wyjsciowa, 'w', encoding='utf-8') as f:
            json.dump(polaczone_dane, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Błąd podczas zapisywania pliku {sciezka_wyjsciowa}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Łączenie plików JSON z annotations i text_marked')
    parser.add_argument('--annotations', type=str, default=FOLDER_ANNOTATIONS,
                        help=f'Ścieżka do folderu z plikami JSON z annotations (domyślnie: {FOLDER_ANNOTATIONS})')
    parser.add_argument('--text_marked', type=str, default=FOLDER_TEXT_MARKED,
                        help=f'Ścieżka do folderu z plikami JSON z text_marked (domyślnie: {FOLDER_TEXT_MARKED})')
    parser.add_argument('--output', type=str, default=FOLDER_COMBINED,
                        help=f'Ścieżka do folderu wyjściowego (domyślnie: {FOLDER_COMBINED})')
    args = parser.parse_args()
    
    # Wczytaj pliki JSON z obu folderów
    pliki_annotations = wczytaj_pliki_json(args.annotations)
    pliki_text_marked = wczytaj_pliki_json(args.text_marked)
    
    # Liczniki dla statystyk
    liczba_par = 0
    liczba_niezgodnych = 0
    liczba_udanych = 0
    
    # Przetwórz każdy plik z annotations
    for sciezka_wzgledna, dane_annotations in pliki_annotations.items():
        # Szukaj odpowiadającego pliku w text_marked
        if sciezka_wzgledna in pliki_text_marked:
            dane_text_marked = pliki_text_marked[sciezka_wzgledna]
            liczba_par += 1
            
            # Porównaj rozmiary obrazów
            info_rozmiary = porownaj_rozmiary_obrazow(dane_annotations, dane_text_marked)
            
            if not info_rozmiary["zgodne_rozmiary"]:
                liczba_niezgodnych += 1
                print(f"Uwaga: Niezgodne rozmiary dla {sciezka_wzgledna}")
                print(f"  Annotations: {info_rozmiary['annotations_size']}")
                print(f"  Text_marked: {info_rozmiary['text_marked_size']}")
                print(f"  Współczynniki skalowania: x={info_rozmiary['skala_x']:.2f}, y={info_rozmiary['skala_y']:.2f}")
            
            # Połącz dane (zawsze, nawet dla niezgodnych rozmiarów)
            polaczone_dane = polacz_pliki_json(dane_annotations, dane_text_marked, info_rozmiary)
            
            # Zapisz połączone dane
            sciezka_wyjsciowa = os.path.join(args.output, sciezka_wzgledna)
            if zapisz_polaczony_plik(polaczone_dane, sciezka_wyjsciowa):
                liczba_udanych += 1
                print(f"Połączono pliki: {sciezka_wzgledna}")
    
    print(f"\nStatystyki:")
    print(f"- Znaleziono {liczba_par} par plików JSON")
    print(f"- Wykryto {liczba_niezgodnych} par z niezgodnymi rozmiarami obrazów (wszystkie zostały doskalowane)")
    print(f"- Pomyślnie połączono i zapisano {liczba_udanych} plików")
    print(f"\nPliki zapisano w folderze: {args.output}")

if __name__ == "__main__":
    main() 