# Detektor Linii w Schematach

Uproszczony projekt służący do wykrywania linii i połączeń między blokami w schematach blokowych.

## Struktura projektu

- `line_detector.py` - zintegrowany moduł do wykrywania linii między blokami
- `main.py` - główny plik programu
- `combined_json/` - katalog z adnotacjami (zaznaczonymi blokami) w formacie JSON
- `results/` - katalog z wynikami (wykryte połączenia)
- `results/debug/` - katalog z obrazami pośrednich etapów algorytmu

## Zastosowane technologie i narzędzia

### Biblioteki i narzędzia

- **Python 3.7+** - język programowania
- **OpenCV (cv2)** - główna biblioteka do przetwarzania obrazów
  - Detekcja krawędzi (Canny)
  - Transformacja Hougha do wykrywania linii prostych
  - Line Segment Detector (LSD) - zaawansowany detektor linii
  - Operacje morfologiczne (dylatacja, erozja)
  - Filtrowanie bilateralne i medianowe do redukcji szumów
- **NumPy** - operacje na macierzach i wektorach
- **Logging** - rejestrowanie zdarzeń i błędów
- **JSON** - obsługa plików konfiguracyjnych i adnotacji

### Techniki przetwarzania obrazów

1. **Wstępne przetwarzanie**:
   - Konwersja do skali szarości
   - Filtrowanie bilateralne (zachowuje krawędzie)
   - Adaptacyjna normalizacja histogramu (CLAHE)
   - Redukcja szumów

2. **Detekcja krawędzi**:
   - Wielopoziomowa detekcja Canny z różnymi progami
   - Łączenie wyników z różnych progów detekcji

3. **Wykrywanie linii**:
   - Priorytetyzacja detektora LSD (Line Segment Detector)
   - Alternatywne wykorzystanie transformacji Hougha
   - Filtrowanie linii według długości, kąta i pozycji

4. **Zaawansowane algorytmy**:
   - Inteligentne łączenie podobnych segmentów linii
   - Usuwanie duplikatów linii
   - Filtrowanie obramowań bloków
   - Wydłużanie linii w kierunku bloków
   - Identyfikacja połączeń między blokami

### Podejście debugowania

- Zapisywanie obrazów pośrednich etapów algorytmu
- Tworzenie osobnych katalogów debug dla każdego schematu
- Szczegółowe logowanie operacji i statystyk

## Wymagania

- Python 3.7+
- OpenCV
- NumPy

## Instalacja

1. Zainstaluj wymagane pakiety:
```bash
pip install -r requirements.txt
```

## Użycie

1. Przygotuj katalogi z adnotacjami `combined_json/Automatyka` i `combined_json/Elektroniczne`

2. Aby wykryć linie między blokami, uruchom:
```bash
python main.py
```

3. Wyniki (obrazy z wykrytymi liniami) zostaną zapisane w katalogu `results/`, a obrazy debug w `results/debug/`

## Najważniejsze funkcje algorytmu

- Wykrywanie linii przy użyciu zaawansowanych algorytmów LSD
- Wielopoziomowa detekcja krawędzi dla lepszej identyfikacji słabych linii
- Inteligentne filtrowanie niepotrzebnych linii (obramowań bloków)
- Identyfikacja połączeń między blokami na podstawie przecięć linii
- Przechodzenie przez wszystkie schematy automatycznie
- Generowanie obrazów debug dla analizy działania algorytmu
