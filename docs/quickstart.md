# Szybki start

Poniżej minimalne kroki, by uruchomić środowisko oraz dokumentację.

## Wymagania

- Python 3.10–3.11
- Windows PowerShell (na Windows) lub bash (Linux/macOS)

## Instalacja

```powershell
# 1) Utwórz i aktywuj wirtualne środowisko
python -m venv .venv; .venv\Scripts\Activate.ps1

# 2) Zainstaluj zależności projektu
pip install -r requirements.txt

# (Opcjonalnie) Zainstaluj dodatkowe zależności do konkretnych modułów, jeśli potrzebne
# np. ultralytics, paddleocr mogą wymagać dodatkowych pakietów systemowych/ML
```

## Dokumentacja lokalnie

Uruchamiaj z katalogu głównego repozytorium (tam, gdzie jest `mkdocs.yaml`).

```powershell
mkdocs serve
```

Następnie otwórz http://127.0.0.1:8000.

## Pierwsze kroki z modułami

- Charts (analiza wykresów): zobacz [Charts »](reference/charts/charts.md)
- OCR: zacznij od [Detekcja elementów »](reference/ocr/scalanie_ocr/detekcja_elementow.md)
- RAG metrics: [BERTscore »](reference/rag_codes/rag_metrics/BERTscore.md)
- Schematics: [Metrics »](reference/schematics/metrics.md)

Przykłady do odpalenia:
- Wygeneruj dane/obrazy testowe: `charts/data_analyzer/example_charts.py`
- Pipeline wykresów: zacznij od modułów w `charts/charts_axes_detect/`
- OCR: skrypty w `ocr/scalanie_ocr/` (sprawdź README i zależności)

## Problemy z importem podczas budowy docs

Jeśli widzisz błędy typu “Could not collect …”:
- uruchamiaj `mkdocs` z katalogu głównego (nie z `docs/`)
- upewnij się, że katalogi są pakietami Pythona (`__init__.py`)
- część ciężkich zależności jest opcjonalna podczas importu (np. wykresy/ML) — sprawdź komunikat
