# Ekstrakcja tekstu z obrazów

## Opis projektu
Projekt służy do wykrywania i ekstrakcji tekstu z obrazów technicznych, takich jak schematy czy dokumentacja elektroniczna i automatyczna.

## Wykorzystywane narzędzia

- **OpenCV** - przetwarzanie obrazów
- **EasyOCR** - rozpoznawanie tekstu
- **Tesseract OCR** - rozpoznawanie tekstu
- **NumPy** - operacje na macierzach
- **Imutils** - pomocnicze funkcje dla OpenCV

## Struktura projektu

- `text_preprocessing.py` - skrypt do wstępnego przetwarzania obrazów
- `text_extraction.py` - skrypt do wykrywania i ekstrakcji tekstu
- `Dataset/` - katalog z obrazami do przetworzenia
- `Wyniki_OCR/` - katalog z wynikami ekstrakcji tekstu

## Instrukcje uruchomienia

### 1. Przygotowanie środowiska

```bash
pip install opencv-python numpy easyocr pytesseract imutils
```

Dodatkowo należy zainstalować Tesseract OCR dla systemu operacyjnego:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt install tesseract-ocr tesseract-ocr-pol`

### 2. Przygotowanie obrazów

Umieść obrazy do analizy w katalogu `Dataset/` w odpowiednich podkatalogach (np. `Dataset/Automatyka/`, `Dataset/Elektroniczne/`).

### 3. Uruchomienie przetwarzania wstępnego

```bash
python text_preprocessing.py --foldery Dataset/Automatyka,Dataset/Elektroniczne
```

### 4. Uruchomienie ekstrakcji tekstu

```bash
python text_extraction.py --foldery Dataset/Automatyka,Dataset/Elektroniczne --metoda hybrid
```

## Parametry

### Przetwarzanie wstępne (text_preprocessing.py):
- `--foldery` - lista folderów z obrazami do przetworzenia
- `--wizualizacja` - flaga do włączenia wizualizacji wyników

### Ekstrakcja tekstu (text_extraction.py):
- `--foldery` - lista folderów z obrazami do przetworzenia
- `--metoda` - metoda OCR (easyocr, tesseract, hybrid)
- `--jezyk` - język tekstu (pol, eng)
- `--prog` - próg pewności dla wykrytego tekstu (0.0-1.0) 