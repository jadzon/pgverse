# Graph - System Zarządzania Grafem Neo4j z Zaawansowanym Uczeniem Się

## Przegląd
Moduł zawiera klasy do zarządzania grafem wiedzy w Neo4j z funkcjonalnościami samouczenia się, automatycznej optymalizacji i śledzenia wzorców użycia. System automatycznie wzmacnia relacje między często używanymi węzłami i dostosowuje się do wzorców użytkowników.

## Klasa Neo4jConnector

### Przegląd
Prosty wrapper dla połączenia z bazą Neo4j z wyłączonym sprawdzaniem wersji serwera.

```python
Neo4jConnector(uri: str, user: str, password: str)
```

### Metody
- `get_driver()` - zwraca driver Neo4j
- `close()` - zamyka połączenie

## Klasa GraphBuilder

### Przegląd
Główna klasa do budowania i utrzymywania samo-uczącej się grafu węzłów tekstowych i multimedialnych. Każdy węzeł może reprezentować tekst, obraz, tabelę lub wzór matematyczny.

### Struktura Węzłów

#### Właściwości Węzła
```python
{
    'id': str,                    # unikalny identyfikator
    'type': str,                  # 'text', 'image', 'table', 'formula' 
    'text': str,                  # treść tekstowa lub opis
    'embedding': List[float],     # reprezentacja wektorowa
    'path': str,                  # ścieżka do pliku zewnętrznego
    'source': str,                # typ źródła z sources_config.json
    'base64': str,                # dane zakodowane w base64 (obrazy/wzory/tabele)
    'usage_count': int,           # liczba użyć
    'created_at': timestamp,      # czas utworzenia
    'last_accessed': timestamp,   # ostatni dostęp
    'embedding_type': str         # 'text_embedding' lub 'image_embedding'
}
```

#### Etykiety Węzłów
- `TextNode` - węzły tekstowe
- `ImageNode` - obrazy i figury  
- `FormulaNode` - wzory matematyczne
- `TableNode` - tabele i dane strukturalne

### Inicjalizacja
```python
GraphBuilder(connector: Neo4jConnector, similarity_threshold: float = 0.95)
```

### Główne Metody

#### `insert_node(node_id, data_type, text, embedding=None, path=None, source="unknown", base64_data=None)`
Wstawia lub aktualizuje węzeł z określonymi właściwościami.

**Parametry:**
- `node_id` - unikalny identyfikator
- `data_type` - typ danych ('text', 'image', 'formula', 'table')
- `text` - treść lub opis
- `embedding` - embedding wektorowy
- `path` - ścieżka do pliku źródłowego
- `source` - typ źródła
- `base64_data` - dane w base64 (tylko dla treści wizualnych)

#### `create_relations()`
Tworzy relacje SIMILAR_TO między węzłami na podstawie podobieństwa cosinusowego embeddingów.

**Algorytm:**
1. Przetwarza każdy typ węzła osobno
2. Oblicza podobieństwo cosinusowe dla wszystkich par
3. Tworzy relacje dla par przekraczających próg
4. Bez limitów - przetwarza wszystkie węzły

**Optymalizacje:**
- Standardowe przetwarzanie dla <200 węzłów
- Przetwarzanie partiami dla 200-1000 węzłów  
- Zaawansowane algorytmy dla >1000 węzłów

#### `create_relations_with_progress_callback(progress_callback=None)`
Wersja z callbackiem do GUI - zoptymalizowana dla dużej liczby węzłów.

**Strategie:**
- **Małe zestawy (≤200)**: standardowe przetwarzanie
- **Średnie (200-1000)**: przetwarzanie partiami
- **Duże (>1000)**: zaawansowane algorytmy z optymalizacjami

### Funkcje Samo-uczenia się

#### `reinforce_relationship(node_a_id, node_b_id, strength=1.0)`
Wzmacnia relację między węzłami na podstawie współwystępowania.

**Mechanizm:**
- Tworzy relację z wagą 0.1 jeśli nie istnieje
- Zwiększa wagę o `strength * 0.1` (max 1.0)
- Zwiększa licznik wzmocnień
- Zapisuje czas ostatniego użycia

#### `track_usage_pattern(node_ids, query_context="")`
Śledzi wzorce użycia - które węzły są często używane razem.

**Funkcjonalności:**
- Zwiększa liczniki użycia dla każdego węzła
- Wzmacnia relacje między wszystkimi parami węzłów
- Zapisuje wzorce lokalnie dla analizy
- Dynamicznie dostosowuje siłę wzmocnień

#### `increment_node_usage(node_id, context="")`
Zwiększa licznik użycia węzła i zapisuje kontekst.

### Analityka i Statystyki

#### `analyze_learning_patterns()`
Analizuje wzorce uczenia się i zwraca rozszerzone statystyki.

**Zwracane dane:**
```python
{
    'node_statistics_by_type': [...]     # statystyki węzłów według typu
    'relation_statistics': [...]         # statystyki relacji
    'source_statistics': [...]           # statystyki źródeł
    'total_statistics': {...}            # ogólne statystyki
    'usage_patterns_count': int,         # liczba wzorców użycia
    'current_threshold': float           # aktualny próg podobieństwa
}
```

### Maintenance i Optymalizacja

#### `decay_relationships(decay_factor=0.95, min_weight=0.1)`
Zmniejsza wagę relacji nieużywanych przez długi czas.

**Strategia:**
- Silny decay (×0.76) dla relacji starszych niż 30 dni
- Łagodny decay (×0.95) dla relacji 7-30 dni
- Usuwa relacje poniżej minimalnej wagi z małą liczbą wzmocnień

#### `adaptive_threshold_adjustment()`
Automatycznie dostosowuje próg podobieństwa na podstawie gęstości grafu.

**Algorytm:**
```python
density = relation_count / max_possible_relations

if density > 0.3:      # za dużo relacji
    threshold += adjustment
elif density < 0.05:   # za mało relacji  
    threshold -= adjustment
```

#### `prune_old(max_age_ms)`
Usuwa stare relacje nieaktualizowane przez określony czas.

**Kryteria usunięcia:**
- Nieużywane dłużej niż `max_age_ms`
- Mniej niż 3 wzmocnienia
- Waga uczenia się < 0.3

#### `run_maintenance()`
Uruchamia pełną konserwację grafu:
1. Zastosowanie decay relacji
2. Adaptacyjna zmiana progów  
3. Usunięcie starych relacji

## Klasa LearningPatternTracker

### Przegląd
Zaawansowany system śledzenia wzorców uczenia się i automatycznej optymalizacji grafu.

### Główne Struktury Danych
```python
self.query_history = []              # historia zapytań
self.co_occurrence_patterns = {}     # wzorce współwystępowania
self.temporal_patterns = {}          # wzorce czasowe
self.semantic_clusters = {}          # klastry semantyczne
```

### Automatyczne Uczenie Się

#### `record_query_session(query, retrieved_nodes, user_feedback=None)`
Zapisuje sesję zapytania i uruchamia automatyczne uczenie się.

**Proces:**
1. Zapisuje sesję w historii
2. Automatyczne uczenie z wzorców pobierania
3. Aktualizacja wzorców czasowych
4. Wzmacnianie relacji współwystępowania
5. Przetwarzanie feedbacku użytkownika (jeśli podany)

#### `_learn_from_retrieval_patterns(query, retrieved_nodes)`
Uczy się z wzorców pobierania - które węzły pojawiają się razem.

**Mechanizm:**
- Zapisuje wszystkie pary węzłów z wyników
- Wzmacnia relacje co 3 współwystąpienia
- Przypisuje słowa kluczowe z zapytań do węzłów

#### `_strengthen_co_occurrence_relations(retrieved_nodes)`
Wzmacnia relacje między węzłami często występującymi razem.

### Odkrywanie Wzorców

#### `discover_semantic_clusters()`
Automatycznie odkrywa klastry semantyczne na podstawie wzorców użycia.

**Algorytm:**
1. Znajdź relacje z wysokimi liczbami wzmocnień
2. Grupuj węzły w klastry na podstawie silnych połączeń
3. Łącz klastry gdy węzły należą do wielu grup
4. Zwraca mapę klastrów

#### `analyze_usage_patterns()`
Analizuje wzorce użycia i identyfikuje trendy.

**Zwracane informacje:**
```python
{
    'popular_nodes': [...],           # najpopularniejsze węzły
    'total_queries': int,             # liczba zapytań
    'feedback_sessions': int,         # sesje z feedbackiem
    'auto_learning_stats': {...}     # statystyki automatycznego uczenia
}
```

### Automatyczna Optymalizacja

#### `auto_optimize_graph()`
Automatyczna optymalizacja grafu na podstawie nauczonych wzorców.

**Optymalizacje:**
1. **Wzmacnianie klastrów** - zwiększa wagi relacji w descobranych klastrach
2. **Identyfikacja hubów** - oznacza często używane węzły jako centralne
3. **Usuwanie słabych relacji** - czyści nieistotne połączenia

**Kryteria hubów:**
- Minimum 5 zapytań
- Minimum 3 połączenia
- Obliczenie hub_score na podstawie użycia i połączeń

## Przykłady Użycia

### Podstawowe Budowanie Grafu
```python
# Połączenie z Neo4j
connector = Neo4jConnector("bolt://localhost:7687", "neo4j", "password")
builder = GraphBuilder(connector, similarity_threshold=0.85)

# Dodanie węzła tekstowego
builder.insert_node(
    node_id="text_001",
    data_type="text", 
    text="To jest przykład tekstu o machine learning",
    embedding=embedding_vector,
    source="wikipedia"
)

# Dodanie węzła obrazu
builder.insert_node(
    node_id="img_001",
    data_type="image",
    text="Diagram sieci neuronowej", 
    embedding=image_embedding,
    path="images/neural_network.png",
    source="książka",
    base64_data=base64_encoded_image
)

# Tworzenie relacji podobieństwa
builder.create_relations()
```

### Z Zaawansowanym Uczeniem Się
```python
connector = Neo4jConnector("bolt://localhost:7687", "neo4j", "password")
learning_tracker = LearningPatternTracker(connector)

# Symulacja sesji zapytania
retrieved_nodes = ["node_1", "node_3", "node_7"]
learning_tracker.record_query_session(
    query="neural networks",
    retrieved_nodes=retrieved_nodes,
    user_feedback={
        'useful_nodes': ["node_1", "node_3"],
        'not_useful_nodes': ["node_7"]
    }
)

# Automatyczna optymalizacja
optimizations = learning_tracker.auto_optimize_graph()
print(f"Zastosowano {len(optimizations)} optymalizacji")

# Analiza wzorców
patterns = learning_tracker.analyze_usage_patterns()
print(f"Najpopularniejszy węzeł: {patterns['popular_nodes'][0]}")
```

### Maintenance i Monitoring
```python
builder = GraphBuilder(connector)

# Regularna konserwacja
builder.run_maintenance()

# Analiza statystyk
stats = builder.analyze_learning_patterns()
print(f"Węzłów z embeddingami: {stats['total_statistics']['nodes_with_embeddings']}")
print(f"Aktualny próg: {stats['current_threshold']}")

# Ręczna optimalizacja progu
builder.adaptive_threshold_adjustment()
```

## Wymagania

### Techniczne
- **Neo4j 4.0+** - baza grafowa
- **Python 3.7+** 
- **neo4j-driver** - sterownik Python dla Neo4j
- **numpy** - operacje numeryczne
- **collections.defaultdict** - struktury danych

### Konfiguracja Neo4j
```cypher
// Tworzenie indeksów dla wydajności
CREATE INDEX node_id_idx IF NOT EXISTS FOR (n:TextNode) ON (n.id);
CREATE INDEX node_embedding_idx IF NOT EXISTS FOR (n:TextNode) ON (n.embedding);
CREATE INDEX image_id_idx IF NOT EXISTS FOR (n:ImageNode) ON (n.id);
```

## Optymalizacje Wydajności

### Strategia Skalowania
- **<200 węzłów**: O(n²) - standardowy algorytm
- **200-1000**: Przetwarzanie partiami - O(n²/batch_size)
- **>1000**: Ograniczone przetwarzanie + optymalizacje pamięci

### Memory Management
- Batch processing dla dużych grafów
- Indeksy Neo4j dla szybkich zapytań
- Cleanup słabych relacji
- Automatic garbage collection wzorców

### Performance Monitoring
```python
# Statystyki wydajności
stats = builder.analyze_learning_patterns()
relation_count = sum(stat['count'] for stat in stats['relation_statistics'])
node_count = stats['total_statistics']['total_nodes']
density = relation_count / (node_count * (node_count - 1) / 2)

print(f"Graf density: {density:.4f}")
print(f"Average relations per node: {relation_count / node_count:.2f}")
```