# Szybki start

Poniżej minimalne kroki, by uruchomić środowisko oraz dokumentację.

## Wymagania

- Python 3.10–3.11
- Windows PowerShell (na Windows) lub bash (Linux/macOS)
- Przygotowanie huggingface (token) oraz akceptacja warunków użycia https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct

## Instalacja

```powershell
# 1) Utwórz i aktywuj wirtualne środowisko
python -m venv .venv; .venv\Scripts\Activate.ps1

# 2) Instalacja pytorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# 3) Instalacja detectron
pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# 4) Zainstaluj pozostałe zależności projektu
pip install -r requirements.txt

# (Opcjonalnie) Zainstaluj dodatkowe zależności do konkretnych modułów, jeśli potrzebne
# np. ultralytics, paddleocr, detectron mogą wymagać dodatkowych pakietów systemowych/ML

# Warto zaznaczyć, że uruchomienie niektórych modułów rozpocznie instalację/pobieranie modeli, co może potrfać do kilkudziesięciu minut
```

## Uruchamianie programu

- uruchomienie main_window.py
- poruszając się po UI można wybrać dany moduł, z którego chcemy skorzystać
- po zakończeniu korzystania z danego modułu i zamknięciu jego okna cofamy się do głównego menu

##

## Pierwsze kroki z modułami

Przykłady do odpalenia:
- Wygeneruj dane/obrazy testowe: `charts/data_analyzer/example_charts.py`
- Pipeline wykresów: zacznij od modułów w `charts/charts_axes_detect/`
- OCR: skrypty w `ocr/scalanie_ocr/` (sprawdź README i zależności)

## Problemy z importem podczas budowy docs

Jeśli widzisz błędy typu “Could not collect …”:
- uruchamiaj `mkdocs` z katalogu głównego (nie z `docs/`)
- upewnij się, że katalogi są pakietami Pythona (`__init__.py`)
- część ciężkich zależności jest opcjonalna podczas importu (np. wykresy/ML) — sprawdź komunikat
