# BERTScore Evaluation Module

## Opis

Moduł do obliczania metryk BERTScore dla oceny jakości generowanych odpowiedzi w systemach NLP. Implementuje funkcję porównywania odpowiedzi modelu z referencjami etalonnymi z zapisem wyników do pliku JSON.

## Funkcjonalności

### Główne możliwości

- **Obliczanie BERTScore** - Precision, Recall i F1 dla języka polskiego
- **Batch processing** - Ocena względem wielu referencji jednocześnie
- **Perzystencja wyników** - Zapis do pliku JSON z zachowaniem historii
- **UTF-8 encoding** - Pełna obsługa znaków polskich

## Wymagania techniczne

### Biblioteki Python

```python
import json
from bert_score import score
import enviromental_variables as ev
```

### Wymagania systemowe

- **bert-score** - Biblioteka do obliczania metryk BERTScore
- **PyTorch** - Backend dla modelu BERT
- **transformers** - Modele językowe (automatycznie pobierane)

## Główna funkcja

### compute_and_save_bertscore()

Centralna funkcja modułu odpowiedzialna za kompletny cykl ewaluacji.

#### Sygnatura
```python
def compute_and_save_bertscore(query, answer, references, output_path=ev.JSON_PATH):
```

#### Parametry

- **query** (str) - Zapytanie/pytanie użytkownika
- **answer** (str) - Odpowiedź generowana przez model
- **references** (list) - Lista referencji etalonnowych
- **output_path** (str, optional) - Ścieżka do pliku JSON (domyślnie z ev.JSON_PATH)

#### Zwracane wartości

Funkcja nie zwraca wartości - zapisuje wyniki bezpośrednio do pliku.

## Algorytm obliczania metryk

### Proces ewaluacji

1. **Iteracja po referencjach** - Każda referencja oceniana oddzielnie
2. **Wywołanie BERTScore** - `score([answer], [ref], lang='pl', verbose=False)`
3. **Ekstrakcja tensorów** - Konwersja P, R, F1 na wartości float
4. **Agregacja wyników** - Zbieranie w strukturę results[]

### Konfiguracja BERTScore

#### Parametry wywołania
```python
P, R, F1 = score([answer], [ref], lang='pl', verbose=False)
```

- **lang='pl'** - Optymalizacja dla języka polskiego
- **verbose=False** - Wyciszenie komunikatów diagnostycznych
- **Lista argumentów** - [answer] i [ref] jako listy jednołementowe

#### Metryki

- **Precision (P)** - Dokładność - ile z wygenerowanych tokenów jest poprawnych
- **Recall (R)** - Kompletność - ile z referencyjnych tokenów zostało wygenerowanych  
- **F1 Score** - Średnia harmoniczna precision i recall

## Struktura danych

### Format wpisu JSON

```json
{
  "query": "Pytanie użytkownika",
  "answer": "Odpowiedź modelu",
  "scores": [
    {
      "reference": "Pierwsza referencja",
      "precision": 0.85,
      "recall": 0.78,
      "f1": 0.81
    },
    {
      "reference": "Druga referencja", 
      "precision": 0.92,
      "recall": 0.88,
      "f1": 0.90
    }
  ]
}
```

### Pola wynikowe

- **reference** - Tekst referencyjny użyty do porównania
- **precision** - Wartość precision z BERTScore (float)
- **recall** - Wartość recall z BERTScore (float)
- **f1** - Wartość F1 z BERTScore (float)

## Zarządzanie plikami

### Obsługa pliku JSON

#### Wczytywanie istniejących danych
```python
try:
    with open(output_path, 'r', encoding='utf-8') as rf:
        data = json.load(rf)
except FileNotFoundError:
    data = []
```

#### Zapis z aktualizacją
```python
data.append(entry)
with open(output_path, 'w', encoding='utf-8') as wf:
    json.dump(data, wf, ensure_ascii=False, indent=2)
```

### Konfiguracja zapisu

- **ensure_ascii=False** - Zachowanie znaków Unicode/polskich
- **indent=2** - Formatowanie dla czytelności
- **encoding='utf-8'** - Pełna obsługa znaków specjalnych

## Obsługa błędów

### FileNotFoundError handling

- Automatyczne tworzenie nowej listy przy braku pliku
- Graceful initialization dla pierwszego uruchomienia
- Nie przerywanie działania przy problemach z I/O

### Konwersja tensorów

```python
'precision': float(P[0].item())
'recall': float(R[0].item()) 
'f1': float(F1[0].item())
```

- **tensor.item()** - Ekstrakcja skalarnej wartości z tensora PyTorch
- **float()** - Zapewnienie typu float dla JSON serialization
- **[0]** - Pobranie pierwszego (i jedynego) elementu z tensora

## Integracja z systemami

### Użycie w aplikacjach RAG

Moduł jest zaprojektowany do integracji z systemami Retrieval-Augmented Generation:

```python
from BERTscore import compute_and_save_bertscore

# Po generacji odpowiedzi przez model
compute_and_save_bertscore(
    query=user_question,
    answer=model_response, 
    references=retrieved_references
)
```

### Konfiguracja przez env variables

- **ev.JSON_PATH** - Domyślna ścieżka do pliku wyników
- Możliwość override przez parameter output_path
- Centralna konfiguracja w pliku environmental_variables

## Zastosowania

### Ewaluacja modeli językowych

- **Porównanie odpowiedzi** - Ocena jakości względem ground truth
- **A/B testing** - Porównanie różnych wersji modeli
- **Quality assurance** - Monitoring jakości w czasie

### Analiza wyników

- **Tracking precision/recall** - Analiza kompromisu między metrykami
- **Benchmarking** - Standardowa ocena systemów NLP
- **Debugging** - Identyfikacja problemów z jakością generacji

## Przykład użycia

```python
# Przykładowe dane
query = "Co to jest sztuczna inteligencja?"
answer = "AI to technologia umożliwiająca maszynom uczenie się i rozwiązywanie problemów."
references = [
    "Sztuczna inteligencja to dziedzina informatyki zajmująca się tworzeniem systemów zdolnych do myślenia.",
    "AI oznacza zdolność maszyn do naśladowania ludzkiej inteligencji i uczenia się."
]

# Obliczenie i zapis metryk
compute_and_save_bertscore(query, answer, references)
# Output: "Zapisano wyniki BERTScore do [ścieżka]"
```