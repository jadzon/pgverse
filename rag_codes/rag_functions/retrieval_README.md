# TextRetriever - System Wyszukiwania Tekstowego z Zaawansowanym Uczeniem Się

## Przegląd
`TextRetriever` to zaawansowany system wyszukiwania tekstowego dla grafów Neo4j, który wykorzystuje embeddingi semantyczne i oferuje opcjonalne funkcje uczenia się na podstawie wzorców użycia.

## Główne Funkcjonalności

### Podstawowe Wyszukiwanie
- **Wyszukiwanie semantyczne** - wykorzystuje embeddingi CLIP do znajdowania podobnych treści
- **Filtrowanie według źródeł** - możliwość ograniczenia wyników do konkretnych typów źródeł
- **Filtrowanie według przedmiotów** - wyszukiwanie w określonych dziedzinach wiedzy
- **Hybrydowy scoring** - łączy podobieństwo semantyczne z wagą wiarygodności źródła

### Zaawansowane Uczenie Się (opcjonalne)
- **Śledzenie wzorców użycia** - automatyczne wykrywanie które węzły są często używane razem
- **Feedbacki użytkownika** - możliwość oznaczania przydatnych i nieprzydatnych wyników
- **Automatyczna optymalizacja** - system uczy się z wzorców i poprawia przyszłe wyszukiwania
- **Analityka** - szczegółowe statystyki dotyczące użycia i efektywności

## Klasa TextRetriever

### Inicjalizacja
```python
TextRetriever(connector: Neo4jConnector, similarity_threshold: float = 0.9, enable_learning: bool = False)
```

### Główne Metody

#### `search(query, top_k, score_threshold, source_filter, subject_filter, user_feedback, track_patterns)`
Główna metoda wyszukiwania z obsługą różnych filtrów i opcjonalnym śledzeniem wzorców.

**Parametry:**
- `query` - zapytanie tekstowe
- `top_k` - liczba wyników (domyślnie 10)
- `score_threshold` - minimalny próg podobieństwa (domyślnie 0.9)
- `source_filter` - lista źródeł do filtrowania ['wikipedia', 'książka', ...]
- `subject_filter` - lista przedmiotów ['matematyka', 'fizyka', ...]
- `user_feedback` - feedback z poprzednich zapytań
- `track_patterns` - czy śledzić wzorce użycia

#### `provide_feedback(query, results, useful_nodes, not_useful_nodes, additional_feedback)`
Umożliwia dostarczenie feedbacku po wyszukiwaniu dla poprawy uczenia się.

#### `get_learning_statistics()`
Zwraca szczegółowe statystyki uczenia się i wzorców użycia.

#### `optimize_graph()`
Ręczne uruchomienie optymalizacji grafu na podstawie nauczonych wzorców.

#### `discover_clusters()`
Ręczne odkrywanie klastrów semantycznych.

#### `run_advanced_analytics()`
Uruchamia wszystkie zaawansowane funkcje analityczne.

## Funkcje Zarządzania Uczeniem Się

### `enable_advanced_learning()` / `disable_advanced_learning()`
Włączanie/wyłączanie zaawansowanych funkcji uczenia się w czasie działania.

## Algorytm Wyszukiwania

1. **Generowanie embeddingu zapytania** - wykorzystuje CLIPEmbedder
2. **Wyszukiwanie w węzłach TextNode** - oblicza podobieństwo cosinusowe
3. **Hybrydowy scoring** - łączy podobieństwo semantyczne z wagą źródła
4. **Wykorzystanie relacji SIMILAR_TO** - wzmacnia wyniki połączonych węzłów
5. **Uczenie się z wzorców** (opcjonalne) - zapisuje sesję i uczy się z feedbacku

## Integracje

- **Neo4jConnector** - połączenie z bazą grafową
- **CLIPEmbedder** - generowanie embeddingów
- **HybridWeightFunction** - obliczanie wag hybrydowych
- **LearningPatternTracker** - zaawansowane uczenie się (opcjonalne)

## Przykład Użycia

```python
# Podstawowe użycie
retriever = TextRetriever(connector, similarity_threshold=0.85)
results = retriever.search("machine learning algorithms", top_k=5)

# Z uczeniem się
retriever = TextRetriever(connector, enable_learning=True)
results = retriever.search("neural networks", source_filter=['wikipedia', 'artykuł_naukowy'])

# Dostarczenie feedbacku
retriever.provide_feedback(
    query="neural networks",
    results=results,
    useful_nodes=['node1', 'node3'],
    not_useful_nodes=['node5']
)

# Analityka
stats = retriever.get_learning_statistics()
print(f"Total queries: {stats['query_statistics']['total_queries']}")
```

## Wymagania

- Neo4j z grafem zawierającym węzły TextNode z embeddingami
- CLIPEmbedder do generowania embeddingów
- HybridWeightFunction do scoring
- LearningPatternTracker (opcjonalnie dla zaawansowanego uczenia się)