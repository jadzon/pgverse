# LaTeX - System Embeddingów Wzorów Matematycznych

## Przegląd
Zaawansowany system do parsowania, przetwarzania i generowania embeddingów dla wzorów matematycznych w formacie LaTeX. Wykorzystuje Graph Neural Networks (GNN), struktury AST oraz integrację z Cohere API dla tworzenia specjalizowanych reprezentacji wektorowych wzorów matematycznych.

## Architektura Systemu

### Główne Komponenty
1. **LaTeXToGraph** - parser LaTeX do struktury grafowej
2. **MathFormulaGNN** - sieć neuronowa dla grafów matematycznych  
3. **CohereMultilingualEmbedder** - wrapper dla Cohere API v4
4. **JSONMathProcessor** - przetwarzanie wzorów z plików JSON
5. **MathGraphEmbeddingSystem** - główny system fuzji embeddingów
6. **FormulaEmbedder** - interfejs wysokopoziomowy

## Klasa LaTeXToGraph

### Przegląd
Parser konwertujący wzory LaTeX na struktury grafowe poprzez AST (Abstract Syntax Tree).

### Funkcjonalności
- **Parsowanie LaTeX** - wykorzystuje SymPy do konwersji LaTeX→AST
- **Budowanie grafu** - tworzy NetworkX graf z AST
- **Obsługa błędów** - fallback strategies dla problematycznych wzorów
- **Obliczanie złożoności** - metryki charakteryzujące wzory

### Główne Metody

#### `parse_latex_to_ast(latex_formula: str)`
Konwertuje LaTeX do AST używając SymPy.

**Proces:**
1. Czyszczenie wzoru z tagów i nadmiarowych znaków
2. Zastępowanie problematycznych konstrukcji (np. `\left|`, `\right|`)
3. Parsowanie przez `parse_latex()` z SymPy
4. Fallback do uproszczonego parsowania przy błędach

**Przykład:**
```python
parser = LaTeXToGraph()
expr = parser.parse_latex_to_ast(r"\frac{x^2 + 1}{x - 2}")
# Zwraca: Add(Mul(Pow(x, 2), 1), Div(1, Add(x, -2)))
```

#### `build_graph_from_ast(expr)`
Buduje graf NetworkX z wyrażenia AST.

**Struktura węzłów:**
```python
{
    'id': int,                    # unikalny identyfikator w grafie
    'type': str,                  # typ węzła AST (Add, Mul, Pow, Symbol, etc.)
    'value': str,                 # string reprezentacja węzła
    'is_leaf': bool,              # czy węzeł jest liściem
    'latex_repr': str,            # reprezentacja LaTeX
    'complexity': int             # złożoność poddrzewa
}
```

**Relacje:**
- `arg_0`, `arg_1`, ... - relacje między węzłami rodzic-dziecko
- Wagi krawędzi: `1.0 / (i + 1)` gdzie i to indeks argumentu

## Klasa MathFormulaGNN

### Przegląd
Graph Neural Network zoptymalizowana dla struktur matematycznych.

### Architektura
```python
MathFormulaGNN(vocab_size: int, embedding_dim: int = 1024, hidden_dim: int = 1024)
```

**Warstwy:**
1. `nn.Embedding` - embedding dla typów węzłów AST
2. `GCNConv` (3 warstwy) - konwolucje grafowe  
3. `nn.Dropout` - regularyzacja
4. `nn.LayerNorm` - normalizacja
5. `global_mean_pool` - agregacja do reprezentacji całego grafu

### Forward Pass
```python
def forward(self, x, edge_index, batch):
    x = self.embedding(x)                    # [nodes, embedding_dim]
    x = torch.relu(self.conv1(x, edge_index))# GCN layer 1
    x = self.dropout(x)
    x = torch.relu(self.conv2(x, edge_index))# GCN layer 2
    x = self.dropout(x)
    x = self.conv3(x, edge_index)            # GCN layer 3 (bez ReLU)
    x = self.layer_norm(x)                   # normalizacja
    graph_embedding = global_mean_pool(x, batch)  # [batch_size, 1024]
    return graph_embedding
```

## Klasa CohereMultilingualEmbedder

### Przegląd
Wrapper dla Cohere API v4 dostosowany do embeddingów matematycznych.

### Aktualizacje v4
- **Model**: `embed-multilingual-v4.0` (poprzednio v3.0)
- **Zwiększone wymiary** - dynamiczne wykrywanie rozmiaru embeddingu
- **Ulepszona wielojęzyczność** - lepsze wsparcie dla formuł z tekstem w różnych językach

### Główne Metody

#### `get_text_embedding(text: str, input_type: str = "search_document")`
Pobiera embedding tekstowy z Cohere v4.

**Funkcjonalności:**
- Walidacja i filtrowanie pustych tekstów
- Fallback do zerowego embeddingu przy błędach
- Automatyczne wykrywanie wymiaru embeddingu v4
- Comprehensive error handling

## Klasa MathGraphEmbeddingSystem

### Przegląd
Główny system łączący strukturalne embeddingi GNN z kontekstowymi embeddingami tekstowymi.

### Architektura Fuzji
```python
# Warstwa fuzji dostosowana do Cohere v4
self.fusion_layer = nn.Sequential(
    nn.Linear(1024 + 128, 512),    # Cohere v4 (1024) + GNN (128)
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(), 
    nn.Linear(256, 128),           # finalny embedding 128D
    nn.Tanh()
)
```

### Główne Metody

#### `create_math_embedding(latex_formula: str, context_text: str = "")`
Tworzy hybrydowy embedding łączący strukturę matematyczną i kontekst tekstowy.

**Proces:**
1. **Structural Embedding:**
   - Parsowanie LaTeX → AST
   - Konwersja AST → NetworkX graf
   - GNN forward pass → 128D embedding
   
2. **Textual Embedding:**
   - Embedding kontekstu przez Cohere v4 → 1024D embedding
   
3. **Fusion:**
   - Konkatenacja embedddingów: [1024D + 128D]
   - Forward pass przez fusion layer → 128D finalny embedding

## Klasa FormulaEmbedder

### Przegląd
Główny interfejs wysokopoziomowy dla użytkowników. Zapewnia proste API dla różnych scenariuszy użycia.

### Główne Metody

#### `get_formula_structural_embedding(latex_formula: str)`
Tworzy embedding tylko strukturalny dla wzoru LaTeX.

**Zastosowania:**
- Porównywanie podobieństwa strukturalnego wzorów
- Klasyfikacja typów równań
- Wyszukiwanie wzorów o podobnej strukturze

#### `get_formula_by_json_path(json_file_path: str, png_filename: str)`
Znajduje wzór LaTeX w pliku JSON i zwraca jego embedding.

**Funkcjonalności:**
- Automatyczne wyszukiwanie plików JSON w strukturze katalogów
- Mapowanie nazw PNG → wzory LaTeX
- Czyszczenie wzorów z artefaktów JSON ($$, \tag{}, etc.)
- Generowanie embedddingów strukturalnych

#### `get_multiple_formulas_from_json(json_file_path: str, png_filenames: List[str])`
Batch processing wielu wzorów z jednego pliku JSON.

### Metody Statyczne

#### `get_formula_embedding_from_paths(json_file_path: str, png_filename: str)`
Statyczna metoda dla użycia bez instancji klasy.

**Funkcjonalności:**
- Kompletna funkcjonalność bez zarządzania stanem
- Idealna do integracji z innymi systemami
- Automatyczne resource management

#### `get_latex_embedding(latex_formula: str)`  
Bezpośrednie tworzenie embeddingu z wzoru LaTeX.

**Zastosowania:**
- Przetwarzanie wzorów bez plików JSON
- Integracja z edytorami matematycznymi
- Real-time analiza wzorów

## Przetwarzanie JSON

### Klasa JSONMathProcessor

#### `detect_json_formulas(json_data)`
Inteligentne wykrywanie wzorów matematycznych w strukturach JSON.

**Strategia wykrywania:**
1. **Klucze sugerujące wzory**: 'formula', 'equation', 'math', 'latex', 'wzor', 'rownanie'
2. **Wzorce matematyczne**:
   ```python
   math_indicators = [
       r'\\frac', r'\\sqrt', r'\\sum', r'\\int', r'\\alpha', r'\\beta',
       r'\$.*\$', r'\\begin{equation}', r'\\begin{align}',
       r'x\^', r'y\^', r'z\^', r'=', r'\+', r'\-', r'\*', r'/'
   ]
   ```
3. **Rekursywne przeszukiwanie** całej struktury JSON

### Format JSON
```json
{
    "formula_001.png": "$$\\frac{x^2 + y^2}{\\sqrt{x^2 - 1}}$$",
    "equation_002.png": "$$E = mc^2 \\tag{1.1}$$",
    "complex_formula.png": "$$\\sum_{i=1}^{n} \\frac{1}{i^2} = \\frac{\\pi^2}{6}$$"
}
```

## Przykłady Użycia

### Podstawowe Embedding Wzoru
```python
# Bezpośrednie użycie
embedding = FormulaEmbedder.get_latex_embedding(r"\frac{x^2 + 1}{x - 2}")
print(f"Embedding shape: {embedding.shape}")  # (512,)

# Z kontekstem
embedder = FormulaEmbedder("cohere_api_key")
embedding = embedder.get_formula_structural_embedding(r"E = mc^2")
```

### Embedding z Pliku JSON
```python
# Pojedynczy wzór
embedding = FormulaEmbedder.get_formula_embedding_from_paths(
    "wzory/latex_wzory.json", 
    "formula_001.png"
)

# Wiele wzorów
embedder = FormulaEmbedder("cohere_api_key")
embeddings = embedder.get_multiple_formulas_from_json(
    "wzory/latex_wzory.json",
    ["formula_001.png", "equation_002.png", "complex_formula.png"]
)

for png_name, embedding in embeddings.items():
    if embedding is not None:
        print(f"{png_name}: embedding shape {embedding.shape}")
```

### Porównywanie Podobieństwa Wzorów
```python
import numpy as np

# Embeddingi dwóch wzorów
emb1 = FormulaEmbedder.get_latex_embedding(r"\frac{x^2}{y}")
emb2 = FormulaEmbedder.get_latex_embedding(r"\frac{a^2}{b}")

# Podobieństwo cosinusowe
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"Structural similarity: {similarity:.4f}")
```

### Integracja z Systemami RAG
```python
def find_similar_formulas(query_latex, formula_database, top_k=5):
    query_emb = FormulaEmbedder.get_latex_embedding(query_latex)
    
    similarities = []
    for formula_name, formula_latex in formula_database.items():
        formula_emb = FormulaEmbedder.get_latex_embedding(formula_latex)
        sim = np.dot(query_emb, formula_emb)
        similarities.append((sim, formula_name, formula_latex))
    
    # Sortuj i zwróć top_k
    similarities.sort(reverse=True)
    return similarities[:top_k]

# Użycie
database = {
    "pythagorean": r"a^2 + b^2 = c^2",
    "quadratic": r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
    "euler": r"e^{i\pi} + 1 = 0"
}

similar = find_similar_formulas(r"x^2 + y^2 = r^2", database)
```

## Funkcje Pomocnicze

### Czyszczenie Wzorów LaTeX
```python
def _clean_latex_formula(self, latex_formula: str) -> str:
    # Usuń zewnętrzne $$
    cleaned = latex_formula.strip()
    if cleaned.startswith('$$') and cleaned.endswith('$$'):
        cleaned = cleaned[2:-2].strip()
    
    # Usuń znaki nowej linii i nadmiarowe spacje
    cleaned = re.sub(r'\n+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Usuń tagi numeracji
    cleaned = re.sub(r'\\tag\{[^}]*\}', '', cleaned)
    
    # Usuń komentarze końcowe (1.23)
    cleaned = re.sub(r'\s*\([0-9.]+\)\s*$', '', cleaned)
    
    return cleaned.strip()
```

### Wyszukiwanie Plików
```python
def _find_file_in_directory(self, filename: str, search_path: str = "."):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None
```

## Wymagania Techniczne

### Python Packages
```
torch>=1.9.0
torch-geometric>=2.0.0
networkx>=2.6.0
sympy>=1.8.0
cohere>=4.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

### Hardware Requirements
- **CPU**: Multi-core (GNN training benefit from parallelization)
- **GPU**: CUDA-compatible (optional, speeds up GNN inference) 
- **RAM**: 8GB+ (for large formula graphs)
- **Storage**: SSD recommended for fast JSON I/O

## Ograniczenia i Rozwiązania

### Parsowanie LaTeX
**Problem**: Niektóre wzory LaTeX mogą być niepoprawnie sparsowane przez SymPy.
**Rozwiązanie**: Multi-stage fallback strategy z uproszczaniem konstrukcji.

### Skalowalność GNN
**Problem**: Duże grafy wzorów mogą wymagać dużo pamięci.
**Rozwiązanie**: Batch processing i optymalizacje PyTorch Geometric.

### Jakość Embeddingów
**Problem**: Embeddingi mogą nie oddawać wszystkich aspektów matematycznych.
**Rozwiązanie**: Fusion z embedddingami tekstowymi Cohere dla kontekstu.

## Rozszerzenia i Rozwój

### Planowane Funkcjonalności
1. **Kategorie wzorów** - automatyczna klasyfikacja (algebra, calculus, etc.)
2. **Semantic search** - wyszukiwanie wzorów według znaczenia matematycznego
3. **Formula completion** - sugerowanie zakończenia częściowych wzorów
4. **Multi-language support** - wzory z tekstem w różnych językach