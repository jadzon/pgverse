# Ekstrakcja tekstu z obrazów

## Opis projektu
Projekt służy do wykrywania i ekstrakcji tekstu z obrazów technicznych, takich jak schematy czy dokumentacja elektroniczna i automatyczna. Głównym elementem projektu jest plik `text_extraction.py`, który realizuje rozpoznawanie tekstu na obrazach z wykorzystaniem technik przetwarzania obrazu i OCR (Optical Character Recognition).

## Główne narzędzia
- **OpenCV** - przetwarzanie obrazów
- **EasyOCR** - rozpoznawanie tekstu
- **NumPy** - operacje na macierzach

## Struktura projektu
- `text_extraction.py` - **główny skrypt** do wykrywania i ekstrakcji tekstu 
- `text_preprocessing.py` - pomocniczy skrypt do wstępnego przetwarzania obrazów
- `Dataset/` - katalog z obrazami do przetworzenia
- `Wyniki_OCR/` - katalog z wynikami ekstrakcji tekstu (wizualizacje)
- `text_marked/` - katalog z wynikami w formacie JSON

## Text Extraction - opis działania

Plik `text_extraction.py` stanowi główny element projektu i odpowiada za wykrywanie i ekstrakcję tekstu z obrazów. 

### Główne funkcjonalności:

1. **Obsługa różnych formatów obrazów** (JPG, PNG)
2. **Wielopoziomowe przetwarzanie obrazu** (binaryzacja, wyrównanie kontrastu, redukcja szumu)
3. **Wykrywanie obszarów tekstu** za pomocą algorytmów MSER i EAST
4. **Rozpoznawanie tekstu** za pomocą EasyOCR
5. **Zapisywanie wyników** w formacie JSON oraz jako wizualizacje
6. **Zwracanie słownika z wynikami** dla dalszego przetwarzania

## Inicjalizacja i użycie text_extraction.py

### Jak zainicjalizować text_extraction.py

#### 1. Jako samodzielny skrypt z wiersza poleceń:

```bash
# Podstawowe wywołanie z domyślnymi parametrami
python text_extraction.py

# Wywołanie z własnymi parametrami
python text_extraction.py --foldery "Dataset/Folder1,Dataset/Folder2" --prog 0.4 --jezyk pl
```

#### 2. Jako moduł importowany w innym skrypcie Python:

```python
# Import modułu
import text_extraction

# Wywołanie funkcji głównej (z domyślnymi parametrami)
wyniki = text_extraction.main()

# Alternatywnie można przekazać argumenty przez sys.argv przed wywołaniem main()
import sys
sys.argv = ['text_extraction.py', '--prog', '0.4', '--jezyk', 'pl'] 
wyniki = text_extraction.main()
```

### Co dokładnie zwraca text_extraction.py

Funkcja `main()` w `text_extraction.py` zwraca słownik (dictionary) zawierający wyniki detekcji tekstu. Słownik ten ma następującą strukturę:

```python
{
    'nazwa_obrazu_1': {                # Klucz to nazwa obrazu (np. "Automatyka_schemat1")
        'image_path': str,             # Ścieżka do oryginalnego obrazu (string)
        'blocks': [                    # Lista wykrytych bloków tekstu
            {   
                'coords': [float, float, float, float],  # [x_min, y_min, x_max, y_max]
                'type': 'rectangle',    # Typ bloku (zawsze "rectangle")
                'text': str,            # Wykryty tekst
                'confidence': float     # Poziom pewności (0.0-1.0)
            },
            # Więcej bloków tekstu...
        ],
        'image_size': {                # Rozmiar oryginalnego obrazu
            'width': int,              # Szerokość w pikselach
            'height': int              # Wysokość w pikselach
        }
    },
    'nazwa_obrazu_2': { /* ... */ },
    # Więcej obrazów...
}
```

### Opis elementów słownika wynikowego

#### Klucze główne:
- **`nazwa_obrazu`** - Identyfikator obrazu utworzony z nazwy folderu i nazwy pliku (np. "Automatyka_schemat1")

#### Wartości dla każdego obrazu:
- **`image_path`** - Względna ścieżka do oryginalnego obrazu
- **`blocks`** - Lista wykrytych bloków tekstu na obrazie
- **`image_size`** - Słownik zawierający wymiary obrazu (szerokość i wysokość)

#### Każdy blok tekstu (`blocks`) zawiera:
- **`coords`** - Lista czterech liczb zmiennoprzecinkowych [x_min, y_min, x_max, y_max] określających położenie prostokąta zawierającego tekst
- **`type`** - Typ bloku (zawsze "rectangle")
- **`text`** - Wykryty tekst jako string
- **`confidence`** - Poziom pewności rozpoznania jako liczba zmiennoprzecinkowa z zakresu 0.0-1.0 (gdzie 1.0 oznacza 100% pewności)

### Przykład użycia zwróconego słownika

```python
import text_extraction

# Uruchomienie ekstrakcji tekstu i pobranie słownika wyników
wyniki = text_extraction.main()

# Dostęp do wszystkich wykrytych tekstów dla konkretnego obrazu
if 'Automatyka_schemat1' in wyniki:
    obraz_dane = wyniki['Automatyka_schemat1']
    print(f"Plik obrazu: {obraz_dane['image_path']}")
    print(f"Wymiary: {obraz_dane['image_size']['width']}x{obraz_dane['image_size']['height']}")
    
    # Wyświetlenie każdego wykrytego tekstu z pozycją i pewnością
    for i, blok in enumerate(obraz_dane['blocks']):
        print(f"Tekst {i+1}: '{blok['text']}'")
        print(f"  Pozycja: ({blok['coords'][0]}, {blok['coords'][1]}) - ({blok['coords'][2]}, {blok['coords'][3]})")
        print(f"  Pewność: {blok['confidence']:.2f}")

# Filtrowanie wyników z wysoką pewnością dla wszystkich obrazów
teksty_pewne = {}
for nazwa_obrazu, dane in wyniki.items():
    pewne_bloki = [blok for blok in dane['blocks'] if blok['confidence'] > 0.8]
    if pewne_bloki:
        teksty_pewne[nazwa_obrazu] = pewne_bloki

print(f"Znaleziono teksty z wysoką pewnością w {len(teksty_pewne)} obrazach")
```

### Inicjalizacja i użycie

#### Jako samodzielny skrypt

```bash
python text_extraction.py
```

Domyślnie program:
- Wykryje foldery z obrazami w katalogu `Dataset/`
- Przetworzy wszystkie obrazy, wykrywając na nich tekst
- Zapisze wizualizacje w folderze `Wyniki_OCR/`
- Zapisze szczegółowe wyniki w formacie JSON w folderze `text_marked/`
- Zwróci słownik z wynikami

#### Jako moduł w innym skrypcie Python

```python
import text_extraction

# Wywołanie głównej funkcji zwraca słownik z wynikami
wyniki = text_extraction.main()

# Dostęp do wyników dla konkretnego obrazu
if 'Automatyka_schemat1' in wyniki:
    bloki_tekstu = wyniki['Automatyka_schemat1']['blocks']
    for blok in bloki_tekstu:
        print(f"Tekst: {blok['text']}, Pewność: {blok['confidence']}")
        print(f"Położenie: {blok['coords']}")
```

### Parametry wiersza poleceń

Skrypt `text_extraction.py` obsługuje następujące parametry:

- `--foldery` - lista folderów z obrazami, oddzielona przecinkami (domyślnie "Dataset/Automatyka,Dataset/Elektroniczne")
- `--jezyk` - język tekstu do wykrycia (domyślnie "pl")
- `--prog` - minimalny próg pewności dla wykrytego tekstu (0.0-1.0, domyślnie 0.2)
- `--typ` - typ przetworzenia obrazu (domyślnie "binaryzacja_ulepszona")

Przykład użycia z parametrami:

```bash
python text_extraction.py --foldery Dataset/Automatyka --prog 0.4 --jezyk pl
```

### Zwracane dane

Funkcja `main()` zwraca słownik zawierający wyniki detekcji tekstu dla wszystkich przetworzonych obrazów:

```python
{
    'nazwa_obrazu_1': {
        'image_path': 'ścieżka/do/obrazu_1.png',
        'blocks': [
            {
                'coords': [x_min, y_min, x_max, y_max],
                'type': 'rectangle',
                'text': 'wykryty_tekst_1',
                'confidence': 0.95
            },
            # więcej bloków...
        ],
        'image_size': {
            'width': 800,
            'height': 600
        }
    },
    # więcej obrazów...
}
```

Struktura słownika:
- **Klucz główny**: nazwa obrazu (z prefixem folderu, np. 'Automatyka_schemat1')
- **Wartości**:
  - `image_path` - ścieżka do oryginalnego obrazu
  - `blocks` - lista wykrytych bloków tekstu
  - `image_size` - wymiary oryginalnego obrazu

- **Każdy blok tekstu zawiera**:
  - `coords` - współrzędne prostokąta [x_min, y_min, x_max, y_max]
  - `type` - typ bloku (rectangle)
  - `text` - wykryty tekst
  - `confidence` - poziom pewności rozpoznania (0.0-1.0)

### Format plików JSON

Pliki JSON zapisywane w folderze `text_marked/` mają dokładnie taką samą strukturę jak elementy zwracanego słownika:

```json
{
  "image_path": "Dataset\\Automatyka\\schemat1.png",
  "blocks": [
    {
      "coords": [100.0, 150.0, 200.0, 170.0],
      "type": "rectangle",
      "text": "wykryty tekst",
      "confidence": 0.95
    }
  ],
  "image_size": {
    "width": 800,
    "height": 600
  }
}
```

### Przykładowe użycie zwróconego słownika

```python
import text_extraction

# Uruchomienie ekstrakcji tekstu i pobranie wyników
wyniki = text_extraction.main()

# Analiza wyników
for nazwa_obrazu, dane in wyniki.items():
    print(f"Obraz: {nazwa_obrazu}")
    print(f"  - Wymiary: {dane['image_size']['width']}x{dane['image_size']['height']}")
    print(f"  - Liczba wykrytych bloków tekstu: {len(dane['blocks'])}")
    
    # Filtrowanie wyników z wysoką pewnością
    pewne_teksty = [blok for blok in dane['blocks'] if blok['confidence'] > 0.8]
    print(f"  - Teksty z wysoką pewnością: {len(pewne_teksty)}")
    
    # Wypisanie wykrytych tekstów
    for i, blok in enumerate(dane['blocks']):
        print(f"    {i+1}. '{blok['text']}' (pewność: {blok['confidence']:.2f})")
```

### Główne techniki przetwarzania obrazu

Skrypt używa wielu zaawansowanych technik przetwarzania obrazu w celu ulepszenia rozpoznawania tekstu:

1. **Konwersja do skali szarości** - uproszczenie obrazu do analizy
2. **Wyrównanie histogramu (CLAHE)** - poprawa kontrastu dla lepszej widoczności tekstu
3. **Redukcja szumu** - usunięcie zakłóceń utrudniających rozpoznawanie
4. **Binaryzacja adaptacyjna** - uzyskanie czarno-białego obrazu z wyraźnym tekstem
5. **Operacje morfologiczne** - poprawa kształtu liter i znaków
6. **Wykrywanie krawędzi** - dodatkowe wsparcie dla identyfikacji tekstu

Dla każdego obrazu program testuje wszystkie powyższe techniki i wybiera tę, która daje najlepsze wyniki rozpoznawania tekstu.

### Instalacja wymaganych bibliotek

Przed uruchomieniem skryptu należy zainstalować wymagane biblioteki:

```bash
pip install opencv-python numpy easyocr imutils
```

## Pozostałe elementy projektu

**`text_preprocessing.py`**: Skrypt do wstępnego przetwarzania obrazów.

**Struktura folderów**:
- `Dataset/` - oryginalne obrazy
- `Wyniki_OCR/` - wizualizacje wyników
- `text_marked/` - wyniki w JSON 