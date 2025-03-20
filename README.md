# Detektor Połączeń w Schematach

Projekt służący do wykrywania połączeń między blokami w schematach blokowych.

## Struktura projektu

- `calibrate.py` - interfejs do ręcznego zaznaczania bloków
- `connection_detector.py` - moduł wykrywania połączeń między blokami
- `main.py` - główny plik programu
- `annotations/` - katalog z adnotacjami (zaznaczonymi blokami)
- `results/` - katalog z wynikami (wykryte połączenia)

## Wymagania

- Python 3.7+
- OpenCV
- NumPy
- Tkinter

## Instalacja

1. Zainstaluj wymagane pakiety:
```bash
pip install -r requirements.txt
```

## Użycie

1. Uruchom program:
```bash
python main.py
```

2. Użyj interfejsu do zaznaczania bloków
3. Program automatycznie wykryje połączenia między zaznaczonymi blokami
4. Wyniki zostaną zapisane w katalogu `results/`
