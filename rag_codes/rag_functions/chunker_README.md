# TextChunker - Semantyczny System Dzielenia Tekstu

## Przegląd
`TextChunker` to zaawansowany system dzielenia długich tekstów na semantycznie spójne fragmenty (chunki) wykorzystujący embeddingi CLIP i algorytmy programowania dynamicznego do optimalnej segmentacji.

## Główne Funkcjonalności

### Semantyczne Chunking
- **Inteligentne dzielenie** - wykorzystuje podobieństwo semantyczne zamiast prostego podziału na długość
- **Zachowanie kontekstu** - minimalizuje utratę znaczenia na granicach chunków
- **Optymalna długość** - respektuje limity tokenów przy maksymalizacji spójności semantycznej
- **Obsługa struktur** - automatyczne wykrywanie i przetwarzanie rozdziałów

### Algorytm Programowania Dynamicznego
- **Optymalizacja globalna** - znajduje optymalny podział dla całego tekstu, nie tylko lokalny
- **Funkcja reward** - maksymalizuje podobieństwo semantyczne wewnątrz chunków
- **Ograniczenia tokenów** - zapewnia, że żaden chunk nie przekroczy limitu

## Klasa TextChunker

### Inicjalizacja
```python
TextChunker(embedding_model=None)
```
- Automatycznie używa singletona CLIPEmbedder
- Możliwość podania własnego modelu embeddingów

## Główne Metody

### `chunk_from_file(file_path, max_tokens=150)`
Dzieli dokument z pliku na semantyczne chunki.

**Parametry:**
- `file_path` - ścieżka do pliku tekstowego
- `max_tokens` - maksymalna liczba tokenów w chunku (domyślnie 150)

**Funkcjonalności:**
- Automatyczne wykrywanie kodowania UTF-8
- Walidacja istnienia pliku
- Zwraca listę chunków tekstowych

### `chunk_text(text, max_tokens=150)`
Dzieli podany tekst na semantyczne chunki.

**Proces przetwarzania:**
1. **Wykrywanie rozdziałów** - rozpoznaje struktury typu "ROZDZIAŁ", "STAVE", "CHAPTER"
2. **Przetwarzanie rozdziałów** - każdy rozdział jest dzielony osobno
3. **Fragmentacja bazowa** - podział na małe fragmenty (~15 tokenów)
4. **Generowanie embeddingów** - wykorzystuje CLIP do stworzenia reprezentacji wektorowych
5. **Obliczanie podobieństwa** - tworzy macierz podobieństwa między fragmentami
6. **Optymalizacja podziału** - programowanie dynamiczne znajduje optymalny podział

## Algorytm Wewnętrzny

### Fragmentacja Bazowa - `_simple_recursive_split(text, target_tokens)`
- Dzieli tekst na zdania używając regex
- Grupuje zdania w fragmenty ~15 tokenów
- Zachowuje granice zdań dla lepszej czytelności

### Generowanie Embeddingów - `_get_embeddings(texts)`
- Wykorzystuje CLIPEmbedder do tworzenia reprezentacji wektorowych
- Zwraca macierz embeddingów dla wszystkich fragmentów
- Obsługuje błędy i fallbacki

### Optymalizacja Podziału - `_find_optimal_chunks(fragments, sim_matrix, max_tokens)`
Algorytm programowania dynamicznego:

```
dp[i] = maksymalny reward dla segmentacji fragmentów i..n-1
reward(i,j) = suma podobieństw między fragmentami w chunku [i,j]
```

**Proces:**
1. Inicjalizacja tablicy DP
2. Dla każdej pozycji i:
   - Sprawdzenie wszystkich możliwych końców chunka j
   - Walidacja limitu tokenów
   - Obliczenie reward dla chunka [i,j]
   - Wybór optymalnego podziału
3. Rekonstrukcja optymalnego podziału

### Funkcja Reward
```python
def chunk_reward(i, j):
    reward = 0
    for p in range(i, j):
        for q in range(p+1, j):
            reward += sim_matrix[p, q]
    return reward
```
- Sumuje podobieństwa między wszystkimi parami fragmentów w chunku
- Preferuje chunki z wysokim podobieństwem wewnętrznym

## Przykłady Użycia

### Podstawowe Dzielenie Pliku
```python
chunker = TextChunker()
chunks = chunker.chunk_from_file("dokument.txt", max_tokens=200)

print(f"Utworzono {len(chunks)} chunków")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk[:100]}...")
```

### Dzielenie Tekstu w Pamięci
```python
text = """
ROZDZIAŁ 1. WPROWADZENIE

Machine learning to dziedzina sztucznej inteligencji...
"""

chunker = TextChunker()
chunks = chunker.chunk_text(text, max_tokens=150)

for chunk in chunks:
    print(f"Długość: {len(chunk.split())} tokenów")
    print(f"Treść: {chunk}")
    print("-" * 50)
```

### Z Własnym Modelem Embeddingów
```python
from embeddings import CLIPEmbedder

embedder = CLIPEmbedder.get_instance()
chunker = TextChunker(embedding_model=embedder)
chunks = chunker.chunk_text(text)
```

## Obsługiwane Struktury Dokumentów

### Rozdziały
- `ROZDZIAŁ [IVXLCDM\d]+\.?` (polskie numery rzymskie i arabskie)
- `STAVE [IVXLCDM\d]+\.?` (norweskie)  
- `CHAPTER [IVXLCDM\d]+\.?` (angielskie)

### Przykład Struktury
```
ROZDZIAŁ I. WPROWADZENIE
Tutaj jest treść pierwszego rozdziału...

ROZDZIAŁ II. METODOLOGIA
Tutaj jest treść drugiego rozdziału...
```

## Zalety Algorytmu

### Semantyczna Spójność
- Chunki zawierają tematycznie powiązane treści
- Minimalizacja utraty kontekstu na granicach
- Lepsze embeddingi dla całych chunków

### Optymalizacja Globalna
- Uwzględnia wpływ każdego podziału na cały dokument
- Unika zachłannych decyzji lokalnych
- Maksymalizuje ogólną jakość podziału

### Elastyczność
- Dostosowywalna długość chunków
- Obsługa różnych typów dokumentów
- Kompatybilność z różnymi modelami embeddingów

## Wymagania Techniczne

- **Python 3.7+**
- **NumPy** - operacje na macierzach
- **scikit-learn** - obliczanie podobieństwa cosinus
- **CLIPEmbedder** - generowanie embeddingów
- **Regex** - przetwarzanie tekstu

## Wydajność

- **Złożoność czasowa**: O(n³) dla algorytmu DP
- **Złożoność pamięciowa**: O(n²) dla macierzy podobieństwa  
- **Optymalizacje**: Używa NumPy dla szybkich operacji na macierzach

## Ograniczenia

- Duże dokumenty mogą wymagać dużo pamięci dla macierzy podobieństwa
- Czas przetwarzania rośnie kwadratowo z liczbą fragmentów bazowych
- Wymaga stabilnego połączenia z modelem embeddingów