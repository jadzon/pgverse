# O projekcie

PGVerse to zestaw modułów do:
- analizy wykresów (wykrywanie osi, interpretacja, ekstrakcja punktów)
- OCR dokumentów i obrazów (EasyOCR, PaddleOCR)
- oceny jakości odpowiedzi systemów RAG (np. BERTScore)
- metryk i przetwarzania schematów technicznych

## Założenia

- Jedno repo – wiele narzędzi, wspólne style docstringów (Google style)
- Dokumentacja generowana automatycznie (mkdocstrings)
- Minimalne wymagania do importu podczas budowy dokumentacji

## Struktura (wysokopoziomowo)

- `charts/` – analiza wykresów (detekcja osi, interpretacja, przetwarzanie)
- `ocr/` – potoki OCR oraz narzędzia towarzyszące
- `rag_metrics/` – metryki jakości (BERTScore i wizualizacje)
- `schematics/` – metryki i przetwarzanie schematów
- `docs/` – dokumentacja MkDocs (Material)

### Przykładowe ścieżki i pliki

- Obrazy wykresów: `charts/charts_examples/`
- Dane testowe CSV: `charts/dane/`
- Przykłady analizy danych: `charts/data_analyzer/example_charts.py`
- OCR: główne skrypty pipeline’u w `ocr/scalanie_ocr/` (opis w README)
- RAG: metryki w `rag_metrics/`


