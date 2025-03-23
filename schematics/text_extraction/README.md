# Ekstrakcja tekstu z obrazów

## Opis projektu
Projekt służy do wykrywania i ekstrakcji tekstu z obrazów technicznych, takich jak schematy czy dokumentacja elektroniczna i automatyczna.

## Wykorzystywane narzędzia

- **OpenCV** - przetwarzanie obrazów
- **EasyOCR** - rozpoznawanie tekstu
- **NumPy** - operacje na macierzach
- **Imutils** - pomocnicze funkcje dla OpenCV

## Struktura projektu

- `text_preprocessing.py` - skrypt do wstępnego przetwarzania obrazów
- `text_extraction.py` - skrypt do wykrywania i ekstrakcji tekstu
- `Dataset/` - katalog z obrazami do przetworzenia
- `Wyniki_OCR/` - katalog z wynikami ekstrakcji tekstu (wizualizacje)
- `text_marked/` - katalog z wynikami w formacie JSON
- `annotations/` - katalog z plikami adnotacji w formacie JSON

## Instrukcje uruchomienia

### 1. Przygotowanie środowiska

```bash
pip install opencv-python numpy easyocr imutils
```

### 2. Przygotowanie obrazów

Umieść obrazy do analizy w katalogu `Dataset/` w odpowiednich podkatalogach (np. `Dataset/Automatyka/`, `Dataset/Elektroniczne/`).

### 3. Uruchomienie przetwarzania wstępnego

```bash
python text_preprocessing.py
```

Program automatycznie wykryje foldery z obrazami w katalogu `Dataset/`.

### 4. Uruchomienie ekstrakcji tekstu

```bash
python text_extraction.py
```

Program automatycznie:
- Wykryje foldery z obrazami
- Wykona rozpoznawanie tekstu za pomocą EasyOCR
- Zapisze wyniki w formacie JSON w folderze `text_marked`
- Zapisze wizualizacje w folderze `Wyniki_OCR`

### 5. Dodatkowe opcje uruchomienia

Jeśli chcesz wskazać konkretne foldery do analizy:

```bash
python text_extraction.py --foldery Dataset/Automatyka,Dataset/Elektroniczne
```

Jeśli chcesz zmienić próg pewności dla wykrytego tekstu:

```bash
python text_extraction.py --prog 0.4
```

## Parametry

### Przetwarzanie wstępne (text_preprocessing.py):
- `--foldery` - lista folderów z obrazami do przetworzenia (puste = automatyczne wykrywanie)
- `--wizualizacja` - flaga do włączenia wizualizacji wyników

### Ekstrakcja tekstu (text_extraction.py):
- `--foldery` - lista folderów z obrazami do przetworzenia (domyślnie "Dataset/Automatyka,Dataset/Elektroniczne")
- `--jezyk` - język tekstu (pl, en - domyślnie pl)
- `--prog` - próg pewności dla wykrytego tekstu (0.0-1.0, domyślnie 0.2)
- `--typ` - typ przetworzenia obrazu do użycia (domyślnie "binaryzacja_ulepszona")
- `--zapisz_json` - flaga włączająca zapisywanie wyników w formacie JSON (domyślnie włączona)

## Format plików JSON

Program zapisuje wyniki rozpoznawania tekstu w formacie JSON w folderze `text_marked`. Każdy plik JSON zawiera informacje o wykrytym tekście, jego położeniu na obrazie oraz poziomie pewności rozpoznania.

Format pliku JSON:

```json
{
  "image_path": "ścieżka/do/obrazu.png",
  "blocks": [
    {
      "coords": [x_min, y_min, x_max, y_max],
      "type": "rectangle",
      "text": "wykryty tekst",
      "confidence": 0.95
    },
    ...
  ],
  "image_size": {
    "width": 800,
    "height": 600
  }
}
```

Gdzie:
- `image_path` - ścieżka do oryginalnego obrazu
- `blocks` - lista wykrytych bloków tekstu
- `coords` - współrzędne prostokąta zawierającego tekst [x_min, y_min, x_max, y_max]
- `type` - typ bloku (zawsze "rectangle")
- `text` - wykryty tekst
- `confidence` - poziom pewności rozpoznania (0.0-1.0)
- `image_size` - rozmiar oryginalnego obrazu

## Wielopoziomowe przetwarzanie obrazu

Program wykorzystuje zaawansowane techniki przetwarzania obrazu w celu poprawy jakości rozpoznawania tekstu:

1. Konwersja do skali szarości
2. Wyrównanie histogramu (CLAHE)
3. Redukcja szumu
4. Binaryzacja adaptacyjna
5. Operacje morfologiczne
6. Detekcja krawędzi

Dla każdego obrazu testowane są różne metody przetwarzania, a wybierana jest ta, która daje najlepsze wyniki rozpoznawania tekstu.

## Wykrywanie obszarów tekstu

Program wykorzystuje dwie metody do wykrywania potencjalnych obszarów zawierających tekst:

1. **MSER (Maximally Stable Extremal Regions)** - metoda wykrywająca regiony o stabilnej intensywności
2. **EAST (Efficient and Accurate Scene Text detector)** - model głębokiego uczenia do wykrywania tekstu (jeśli dostępny)

## Wizualizacja wyników

Wyniki rozpoznawania tekstu są wizualizowane na oryginalnych obrazach i zapisywane w folderze `Wyniki_OCR`. Wykryty tekst jest oznaczany prostokątami, których kolor zależy od poziomu pewności rozpoznania (od czerwonego dla niskiej pewności do zielonego dla wysokiej). 