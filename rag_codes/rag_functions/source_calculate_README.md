# HybridWeightFunction - Hybrydowa Funkcja Wagowa dla Systemu RAG

## Przegląd
`HybridWeightFunction` implementuje zaawansowaną funkcję wagową, która łączy podobieństwo semantyczne embeddingów z oceną wiarygodności źródła danych, co pozwala na bardziej precyzyjne rankingi w systemach RAG (Retrieval-Augmented Generation).

## Główne Funkcjonalności

### System Wagowy
- **Podobieństwo semantyczne** - wykorzystuje embeddingi do oceny semantycznej zgodności
- **Wiarygodność źródła** - ocenia jakość źródła na podstawie predefiniowanych poziomów zaufania
- **Elastyczna konfiguracja wag** - możliwość dostosowania proporcji między embeddingami a źródłem
- **Automatyczne mapowanie źródeł** - inteligentne dopasowywanie typów źródeł

## Klasa HybridWeightFunction

### Inicjalizacja
```python
HybridWeightFunction(
    embedding_weight: float = 0.7,
    source_weight: float = 0.3,
    source_trust_levels: Optional[Dict[str, float]] = None
)
```

**Parametry:**
- `embedding_weight` - współczynnik wagi dla podobieństwa embeddingów (domyślnie 0.7)
- `source_weight` - współczynnik wagi dla wiarygodności źródła (domyślnie 0.3)
- `source_trust_levels` - słownik mapujący typy źródeł na poziomy zaufania (0.0-1.0)

### Domyślne Poziomy Zaufania Źródeł
```
- artykuł_naukowy: 0.95 (najwyższa wiarygodność)
- książka: 0.9
- wikipedia: 0.85
- news: 0.6
- blog: 0.5
- unknown: 0.4
- forum: 0.3
- social_media: 0.2 (najniższa wiarygodność)
```

## Główne Metody

### `get_source_trust(source_type: str) -> float`
Pobiera poziom zaufania dla danego typu źródła.

**Funkcjonalności:**
- Normalizacja tekstu źródła do małych liter
- Inteligentne dopasowywanie częściowe (np. "blog_osobisty" → "blog")
- Fallback do "unknown" dla nierozpoznanych źródeł

### `calculate_weight(embedding_similarity: float, source_type: str) -> float`
Oblicza hybrydową wagę łączącą podobieństwo embeddingów i wiarygodność źródła.

**Formuła:**
```
hybrid_weight = (embedding_weight × embedding_similarity) + (source_weight × source_trust)
```

**Parametry:**
- `embedding_similarity` - podobieństwo kosynusowe embeddingów (0.0-1.0)
- `source_type` - typ źródła (np. "wikipedia", "blog")

**Zwraca:** Wynikową wagę hybrydową (0.0-1.0)

### `rerank_results(query_results: List[Dict], default_source_type: str = "unknown") -> List[Dict]`
Przelicza wagi wyników wyszukiwania i sortuje je według nowej wagi hybrydowej.

**Funkcjonalności:**
- Automatyczne obliczenie hybrid_score dla każdego wyniku
- Sortowanie wyników według nowej wagi (malejąco)
- Obsługa brakujących informacji o źródle

## Przykłady Użycia

### Podstawowe Użycie
```python
# Inicjalizacja z domyślnymi ustawieniami
hybrid_scorer = HybridWeightFunction()

# Obliczenie wagi dla wyniku z Wikipedii
weight = hybrid_scorer.calculate_weight(
    embedding_similarity=0.85,
    source_type="wikipedia"
)
# Wynik: 0.7 * 0.85 + 0.3 * 0.85 = 0.85

# Obliczenie wagi dla wyniku z bloga
weight = hybrid_scorer.calculate_weight(
    embedding_similarity=0.85,
    source_type="blog"
)
# Wynik: 0.7 * 0.85 + 0.3 * 0.5 = 0.745
```

### Dostosowane Poziomy Zaufania
```python
# Własne poziomy zaufania
custom_trust_levels = {
    "internal_docs": 0.98,
    "external_api": 0.75,
    "user_generated": 0.3
}

hybrid_scorer = HybridWeightFunction(
    embedding_weight=0.8,
    source_weight=0.2,
    source_trust_levels=custom_trust_levels
)
```

### Rerankowanie Wyników
```python
# Lista wyników wyszukiwania
results = [
    {"id": "1", "text": "...", "score": 0.9, "source_type": "wikipedia"},
    {"id": "2", "text": "...", "score": 0.95, "source_type": "blog"},
    {"id": "3", "text": "...", "score": 0.8, "source_type": "artykuł_naukowy"}
]

# Rerankowanie
reranked = hybrid_scorer.rerank_results(results)

# Wyniki posortowane według hybrid_score
for result in reranked:
    print(f"ID: {result['id']}, Hybrid Score: {result['hybrid_score']:.3f}")
```

## Zastosowania

### W Systemach RAG
- Poprawa jakości wyszukiwania poprzez uwzględnienie wiarygodności źródła
- Balansowanie między semantyczną reletywarnością a jakością informacji
- Automatyczne preferowanie bardziej wiarygodnych źródeł przy podobnym podobieństwie semantycznym

### Integracja z TextRetriever
```python
# W klasie TextRetriever
self.hybrid_scorer = HybridWeightFunction(
    embedding_weight=0.7,
    source_weight=0.3
)

# Użycie podczas wyszukiwania
hybrid_score = self.hybrid_scorer.calculate_weight(
    embedding_similarity=cosine_score,
    source_type=source_type
)
```

## Zalety

1. **Elastyczność** - łatwe dostosowanie wag według potrzeb
2. **Skalowalność** - możliwość dodawania nowych typów źródeł
3. **Przejrzystość** - jasny algorytm obliczania wag
4. **Uniwersalność** - działa z różnymi systemami embeddingów
5. **Konfigurowalność** - pełna kontrola nad poziomami zaufania

## Wymagania

- Python 3.7+
- Typing support dla typów generycznych