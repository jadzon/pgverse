# Metadata Context - Zaawansowany System Przetwarzania Obrazów z Kontekstem

## Przegląd
Kompleksowy system do przetwarzania plików tekstowych zawierających obrazy, wzory matematyczne i tabele. Oferuje inteligentne łączenie treści multimedialnych z kontekstem tekstowym przy użyciu embeddingów semantycznych, specjalizowanych embeddingów LaTeX dla wzorów matematycznych oraz zaawansowanych algorytmów dopasowywania.

## Architektura Systemu

### Główne Komponenty
1. **ImageTextProcessor** - główna klasa do przetwarzania plików z obrazami
2. **ImageContextFilter** - filtrowanie i optymalizacja kontekstu obrazów
3. **Integracja z FormulaEmbedder** - specjalne przetwarzanie wzorów matematycznych
4. **System Base64** - konwersja obrazów do formatu osadzonego
5. **Intelligent Path Resolution** - zaawansowane rozwiązywanie ścieżek

## Klasa ImageTextProcessor

### Przegląd
Główna klasa systemu odpowiedzialna za kompleksowe przetwarzanie plików tekstowych z osadzonymi elementami multimedialnymi.

### Inicjalizacja
```python
ImageTextProcessor(chunker=None, max_tokens=150)
```
- **chunker**: Instancja TextChunker (używa singletona jeśli None)
- **max_tokens**: Maksymalna długość chunków tekstowych
- **Lazy loading**: CLIPEmbedder ładowany tylko gdy potrzebny

### Algoritm Przetwarzania

#### `process_file(file_path)` / `process_text(content)`
Implementuje zaawansowany algorytm przetwarzania:

1. **Wykrywanie elementów multimedialnych** - regex pattern `<img src="(path)">`
2. **Segmentacja tekstu** - dzieli tekst na fragmenty przed każdym obrazem
3. **Chunking semantyczny** - wykorzystuje TextChunker dla spójności
4. **Zachowanie kolejności** - przeplata chunki tekstowe z obrazami
5. **Czyszczenie i normalizacja** - usuwa puste fragmenty

**Pattern rozpoznawania:**
```python
image_pattern = r'<img[^>]+src="([^"]+)"[^>]*>'
```

### Klasyfikacja Typów Elementów

#### `get_element_type(element)`
Inteligentne rozpoznawanie typu elementu multimedialnego:

**Hierarchia klasyfikacji:**
1. **Wzory matematyczne** (priority: highest)
   - Keywords: 'wzory', 'formula', 'formulas'
   - Type: 'formula'

2. **Tabele** (priority: medium)
   - Keywords: 'tabele', 'table', 'tables' 
   - Type: 'table'

3. **Obrazy/Figury** (priority: low)
   - Keywords: 'figury', 'figure', 'figures', 'image', 'images', 'obrazy'
   - Type: 'image'

4. **Fallback**: 'image' dla nierozpoznanych tagów `<img>`

### Zaawansowane Embeddingowe Kojarzenie Kontekstu

#### `get_images_with_context_json(texts, selected_subjects, use_vision)`
Zwraca JSON z obrazami i ich optymalnym kontekstem tekstowym.

**Dwie strategie:**
- `use_vision=False`: Embedding-based matching (implementowane)
- `use_vision=True`: Vision-based matching (placeholder)

#### `_get_images_with_context_embedding_json(texts)`
Główny algorytm kojarzenia na podstawie embeddingów.

**Proces:**
1. **Identyfikacja chunków tekstowych** z tablicy wyników
2. **Dla każdego elementu multimedialnego:**
   - Sprawdzenie istnienia pliku
   - Zbieranie potencjalnych chunków kontekstu
   - Wybór optymalnego chunka przez embeddingi
3. **Zwrot struktury JSON**: `[{"path": ["best_chunk"]}]`

### Strategia Zbierania Kontekstu

**Hierarchia źródeł kontekstu:**
1. **Chunk bezpośrednio przed obrazem** (highest priority)
2. **Chunk bezpośrednio po obrazie** (high priority)  
3. **Najbliższy chunk tekstowy** (fallback)

```python
# Chunk przed obrazem
if i > 0 and self.get_element_type(texts[i-1]) == 'text':
    potential_chunks.append(texts[i-1])

# Chunk po obrazie  
if i < len(texts) - 1 and self.get_element_type(texts[i+1]) == 'text':
    potential_chunks.append(texts[i+1])

# Najbliższy chunk (fallback)
if not potential_chunks:
    closest_chunk = self._find_closest_text_chunk(i, text_positions, texts, text_chunks)
```

## Specjalizowane Embeddingowe Dopasowywanie

### Wzory Matematyczne - `_select_best_chunk_for_formula`

**Zaawansowana strategia dla wzorów:**
1. **Detekcja typu**: Rozpoznawanie wzorów przez ścieżkę/nazwę pliku
2. **Lokalizacja JSON**: Inteligentne wyszukiwanie `latex_wzory.json`
3. **Embedding LaTeX**: Wykorzystanie FormulaEmbedder.get_formula_embedding_from_paths
4. **Porównanie z tekstem**: Embedding wzoru vs. CLIP embeddingi tekstu
5. **Normalizacja wymiarów**: Dopasowanie 512D wzoru do 1024D tekstu
6. **Fallback do CLIP**: Przy braku embeddingów LaTeX

**Lokalizacje JSON (sprawdzane sekwencyjnie):**
```python
possible_json_paths = [
    parent_folder / "wzory" / "latex_wzory.json",
    parent_folder / "rezultaty" / "wzory" / "latex_wzory.json", 
    image_dir / "latex_wzory.json",
    image_dir.parent / "latex_wzory.json"
]
```

### Standardowe Obrazy - `_select_best_chunk_by_embedding`

**CLIP-based matching:**
1. **Image embedding**: CLIP embedding dla obrazu
2. **Text embeddings**: CLIP embeddingi dla wszystkich chunków
3. **Cosine similarity**: Obliczenie podobieństwa dla każdej pary
4. **Best match selection**: Wybór chunka z najwyższym podobieństwem

```python
def _calculate_cosine_similarity(self, embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 0.0
    return cosine_similarity([embedding1], [embedding2])[0][0]
```

## Inteligentne Rozwiązywanie Ścieżek

### `_resolve_image_path(image_path)`
Zaawansowany system lokalizacji plików w złożonych strukturach katalogów.

**Strategia wielopoziomowa:**
1. **Absolute path check** - bezpośrednie sprawdzenie
2. **Relative to file directory** - względem folderu z plikiem TXT
3. **Results structure search** - przeszukiwanie rezultaty/{figury,wzory,tabele}
4. **Recursive search** - rekursywne przeszukiwanie podfolderów
5. **pgverse base resolution** - rozwiązywanie względem głównego folderu
6. **Normalization** - obsługa różnych separatorów (`\` vs `/`)

**Obsługiwane struktury:**
```
project/
├── detekcje/
│   └── document.txt
├── rezultaty/
│   ├── figury/
│   ├── wzory/ 
│   └── tabele/
└── pgverse/
    └── images/
```

## System Konwersji Base64

### `image_to_base64(image_path)`
Konwersja obrazów do formatu base64 z obsługą różnych formatów.

**Funkcjonalności:**
- **Format normalization**: Konwersja RGBA/LA → RGB z białym tłem
- **JPEG compression**: Optymalizacja rozmiaru (quality=85)
- **Error handling**: Graceful degradation przy błędach
- **Memory management**: Buforowane operacje I/O

### `create_output_txt_with_base64(texts, output_file)`
Tworzenie pliku wyjściowego z osadzonymi obrazami w base64.

**Format wyjściowy:**
```
Text chunk 1

<base64_image>iVBORw0KGgoAAAANSUhEUgAA...</base64_image>

Text chunk 2  

<base64_image>iVBORw0KGgoAAAANSUhEUgAA...</base64_image>

Text chunk 3
```

**Funkcjonalności:**
- Pomijanie nieistniejących obrazów
- Fallback do komentarzy przy błędach konwersji
- Zachowanie kolejności z oryginalnego przetwarzania
- Obsługa wszystkich typów elementów multimedialnych

## Klasa ImageContextFilter  

### Przegląd
Wyspecjalizowana klasa do filtrowania i optymalizacji kontekstu obrazów w istniejących strukturach JSON.

### Główne Funkcjonalności

#### `filter_image_context(image_path, context_chunks)`
Filtruje kontekst obrazu używając odpowiednich embeddingów.

**Automatyczna detekcja typu:**
```python
if 'wzory' in image_path.lower() or 'formula' in image_path.lower():
    return self._filter_formula_context(image_path, valid_chunks)
else:
    return self._filter_image_context_clip(image_path, valid_chunks)
```

#### `process_images_context(json_data)`
Przetwarza wszystkie obrazy z JSON i filtruje ich kontekst.

**Proces:**
1. Sprawdzenie istnienia każdego obrazu
2. Rozwiązywanie ścieżek dla nieabsolutnych lokalizacji
3. Filtrowanie kontekstu przez odpowiedni algorytm
4. Zwrot zoptymalizowanych danych JSON

## Przykłady Użycia

### Podstawowe Przetwarzanie Pliku
```python
processor = ImageTextProcessor()

# Przetworzenie pliku z obrazami
texts = processor.process_file("document_with_images.txt")

# Analiza wyników
for i, element in enumerate(texts):
    elem_type = processor.get_element_type(element)
    if elem_type == 'text':
        print(f"Chunk {i}: {element[:50]}...")
    else:
        image_path = processor.get_image_path(element)
        print(f"Image {i}: {image_path} (type: {elem_type})")
```

### Generowanie JSON z Kontekstem
```python
processor = ImageTextProcessor()
texts = processor.process_file("document.txt")

# Generuj JSON z optymalnym kontekstem
json_data = processor.get_images_with_context_json(
    texts, 
    selected_subjects=["mathematics", "physics"], 
    use_vision=False
)

# Wynik: [{"path/to/image.png": ["optimal text chunk"]}]
for item in json_data:
    for image_path, context_list in item.items():
        print(f"Image: {image_path}")
        print(f"Best context: {context_list[0][:100]}...")
```

### Tworzenie Pliku z Base64
```python
processor = ImageTextProcessor()
texts = processor.process_file("input.txt")

# Utwórz plik z obrazami jako base64
output_path = processor.create_output_txt_with_base64(
    texts, 
    "output_with_base64.txt"
)

print(f"Created: {output_path}")
```

### Filtrowanie Kontekstu
```python
filter_system = ImageContextFilter()

# Wczytaj dane JSON
json_data = filter_system.load_images_context("images_context.json") 

# Przefiltruj kontekst dla lepszej jakości
filtered_data = filter_system.process_images_context(json_data)

# Wynik: Każdy obraz ma tylko najlepszy chunk kontekstu
```

### Przetwarzanie Embeddingów dla Różnych Typów

```python
processor = ImageTextProcessor()

# Embedding dla tekstu
text_emb = processor.process_text_embedding("Machine learning algorithms")

# Embedding dla obrazu  
image_emb = processor.process_image_embedding('<img src="neural_network.png">')

# Embedding dla wzoru (z zewnętrznym embeddingiem)
formula_emb = processor.process_formula_embedding(
    '<img src="euler_equation.png">', 
    external_embedding=custom_latex_embedding
)

# Embedding dla tabeli
table_emb = processor.process_table_embedding(
    '<img src="data_table.png">',
    external_embedding=table_structure_embedding
)
```

## Konfiguracja i Wymagania

### Struktura Projektu
```
project_root/
├── detekcje/
│   ├── chapter1.txt       # pliki źródłowe z tagami <img>
│   └── chapter2.txt
├── rezultaty/
│   ├── figury/           # obrazy figure  
│   ├── wzory/            # obrazy wzorów + latex_wzory.json
│   └── tabele/           # obrazy tabel
├── wzory/
│   └── latex_wzory.json  # mapowanie PNG → LaTeX
└── rag_functions/
    └── metadata_context.py
```

### Wymagania Techniczne

**Python Packages:**
```
pillow>=8.0.0           # manipulacja obrazów
scikit-learn>=1.0.0     # cosine similarity
numpy>=1.21.0           # operacje numeryczne  
pathlib                 # zaawansowane ścieżki
base64                  # kodowanie obrazów
io                      # operacje buforowe
re                      # regex processing
os                      # system operations
```

**Integracje:**
- `TextChunker` - semantyczne dzielenie tekstu
- `CLIPEmbedder` - embeddingi tekstowe i wizualne  
- `FormulaEmbedder` - specjalne embeddingi LaTeX
- Neo4j GraphBuilder (opcjonalnie)

### Format Pliku Wejściowego
```html
Wprowadzenie do machine learning.

<img src="figury/neural_network_diagram.png">

Sieci neuronowe są kluczowym elementem...

<img src="wzory/backpropagation_formula.png">

Algorytm backpropagation wykorzystuje...

<img src="tabele/training_results.png">

Wyniki eksperymentów pokazują...
```

## Optymalizacje i Performance

### Memory Management
- **Lazy loading**: Embedder ładowany tylko gdy potrzebny
- **Singleton patterns**: Współdzielenie instancji między komponentami
- **Stream processing**: Przetwarzanie obrazów bez ładowania wszystkich do pamięci
- **Garbage collection**: Automatyczne czyszczenie po operacjach

### Caching Strategies  
- **Path resolution cache**: Zapamiętywanie rozwiązanych ścieżek
- **Embedding cache**: Możliwość cache'owania embeddingów (nie implementowane)
- **File existence cache**: Optymalizacja sprawdzania istnienia plików

### Error Handling
```python
# Graceful degradation przy błędach embeddingu
if embedding is None:
    return chunks[0] if chunks else None

# Fallback do CLIP przy problemach z LaTeX
if formula_embedding is None:
    return self._fallback_to_clip_for_formula(image_path, chunks)

# Pomijanie nieistniejących plików
if not os.path.exists(absolute_path):
    print(f"Skipping non-existent image: {image_path}")
    continue
```

## Rozszerzenia i Integracje

### Integracja z Systemami RAG
```python
# Przykład integracji z retrieval system
def create_multimodal_nodes(processor, file_path):
    texts = processor.process_file(file_path)
    nodes = []
    
    for element in texts:
        elem_type = processor.get_element_type(element)
        
        if elem_type == 'text':
            embedding = processor.process_text_embedding(element)
            nodes.append({
                'type': 'text',
                'content': element,
                'embedding': embedding
            })
        elif elem_type in ['image', 'formula', 'table']:
            path = processor.get_image_path(element) 
            embedding = processor.process_image_embedding(element)
            base64_data = processor.image_to_base64(path)
            
            nodes.append({
                'type': elem_type,
                'path': path,
                'embedding': embedding,
                'base64': base64_data
            })
    
    return nodes
```

### Vision API Integration (Placeholder)
```python
# Przygotowane do integracji z Vision API
def _get_images_with_context_vision_json(self, selected_subjects):
    # Placeholder for Vision API integration
    # Could use GPT-4V, Google Vision, etc.
    pass
```