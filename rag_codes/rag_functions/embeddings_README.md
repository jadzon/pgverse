# Embeddings - System Embeddingów Tekstowych i Wizualnych

## Przegląd
Moduł zawiera dwie główne klasy do generowania embeddingów: `CohereEmbedder` dla embeddingów tekstowych przez API Cohere oraz `CLIPEmbedder` dla embeddingów tekstowych i wizualnych przy użyciu modelu CLIP. Obie klasy implementują wzorzec Singleton dla efektywnego zarządzania zasobami.

## Klasa CohereEmbedder

### Przegląd
Singleton wrapper dla Cohere API v4 z automatycznym przełączaniem kluczy API przy błędach i obsługą limitów quota.

### Funkcjonalności
- **Automatyczne przełączanie kluczy** - przy błędach quota próbuje kolejnych kluczy API
- **Obsługa limitów** - inteligentne wykrywanie błędów związanych z limitami użycia
- **Singleton pattern** - jedna instancja na całą aplikację
- **Cohere v4 compatibility** - wykorzystuje najnowsze API embed-multilingual-v4.0

### Inicjalizacja
```python
CohereEmbedder(api_key=None)
```
- Automatycznie wczytuje klucze z pliku `api_keys.json`
- Obsługuje pojedynczy klucz lub listę kluczy
- Implementuje pattern Singleton

### Struktura Pliku Kluczy
```json
{
    "cohere_keys": [
        "klucz_api_1",
        "klucz_api_2", 
        "klucz_api_3"
    ]
}
```
lub prosta lista:
```json
[
    "klucz_api_1",
    "klucz_api_2"
]
```

### Główne Metody

#### `get_text_embedding(text, model="embed-v4.0", input_type="search_document")`
Generuje embedding dla tekstu z automatycznym przełączaniem kluczy.

**Funkcjonalności:**
- Automatyczne skracanie tekstu powyżej 8000 znaków
- Rotacja kluczy API przy błędach quota
- Obsługa błędów i fallbacków
- Zwraca numpy array lub None

## Klasa CLIPEmbedder

### Przegląd
Zaawansowany Singleton wrapper dla modelu CLIP (openai/clip-vit-large-patch14) z automatyczną konwersją wymiarów, wykrywaniem GPU i obsługą embeddingów tekstowych i wizualnych.

### Główne Funkcjonalności
- **Automatyczne wykrywanie CUDA** - wykorzystuje GPU gdy dostępne, fallback na CPU
- **Konwersja wymiarów** - z natywnych 768 wymiarów na 1024 przez warstwę neuronową
- **Dual modality** - embeddingi tekstu i obrazów w tej samej przestrzeni wektorowej
- **Inteligentne ścieżki** - automatyczne rozwiązywanie ścieżek względem struktury projektu
- **Singleton pattern** - jedna instancja z załadowanym modelem

### Specyfikacje Techniczne
- **Model**: openai/clip-vit-large-patch14
- **Wymiar wejściowy**: 768
- **Wymiar wyjściowy**: 1024 (po konwersji)
- **GPU support**: CUDA z automatycznym fallback
- **Normalizacja**: L2 normalization dla wszystkich embeddingów

### Inicjalizacja
```python
CLIPEmbedder()  # Automatyczny Singleton
# lub
embedder = CLIPEmbedder.get_instance()
```

### Automatyczne Wykrywanie GPU
```
🎮 Wykryto 1 GPU
 GPU 0: NVIDIA GeForce RTX 3080 (10.0 GB)
✅ Wybrano urządzenie: cuda:0
```

### Główne Metody

#### `get_text_embedding(text) -> np.ndarray`
Generuje 1024-wymiarowy embedding dla tekstu.

**Proces:**
1. Walidacja i preprocessing tekstu
2. Tokenizacja przez CLIP processor
3. Przeniesienie na GPU/CPU
4. Generowanie embeddingu (768 dim)
5. Konwersja do 1024 wymiarów przez warstwę neuronową
6. L2 normalizacja
7. Zwrot jako numpy array

#### `get_image_embedding(image_path) -> np.ndarray`
Generuje 1024-wymiarowy embedding dla obrazu.

**Funkcjonalności:**
- Automatyczne naprawianie ścieżek (`fix_image_path`)
- Obsługa różnych formatów obrazów
- Preprocessing przez CLIP processor
- Konwersja wymiarów 768→1024
- L2 normalizacja

#### `fix_image_path(image_path) -> str`
Inteligentne rozwiązywanie ścieżek obrazów.

**Obsługiwane scenariusze:**
- Ścieżki absolutne
- Ścieżki względne względem projektu
- Ścieżki zaczynające się od "pgverse"
- Automatyczne przeszukiwanie struktur katalogów
- Fallbacki dla różnych konfiguracji

### Konwersja Wymiarów

#### Warstwa Konwersji
```python
self.dimension_converter = torch.nn.Linear(768, 1024, bias=False)
torch.nn.init.xavier_uniform_(self.dimension_converter.weight)
```

#### Proces `_convert_to_1024_dims`
1. Konwersja do tensor PyTorch
2. Dodanie batch dimension jeśli potrzebne
3. Forward pass przez warstwę Linear
4. L2 normalizacja
5. Konwersja z powrotem do numpy
6. Fallback z padding zerami przy błędach

## Przykłady Użycia

### CohereEmbedder
```python
# Singleton - automatyczne ładowanie
embedder = CohereEmbedder.get_instance()

# Embedding tekstu
text_emb = embedder.get_text_embedding("Przykładowy tekst do embeddingu")
print(f"Wymiar embeddingu: {text_emb.shape}")  # (1024,)

# Przy błędzie quota automatycznie przełączy klucz
```

### CLIPEmbedder
```python
# Singleton
clip = CLIPEmbedder.get_instance()

# Embedding tekstu
text_emb = clip.get_text_embedding("Machine learning algorithms")
print(f"Text embedding shape: {text_emb.shape}")  # (1024,)

# Embedding obrazu
image_emb = clip.get_image_embedding("path/to/image.jpg") 
print(f"Image embedding shape: {image_emb.shape}")  # (1024,)

# Obliczenie podobieństwa
similarity = np.dot(text_emb, image_emb)
print(f"Similarity: {similarity}")
```

### Integracja Multimodalna
```python
def find_best_image_for_text(text, image_paths):
    clip = CLIPEmbedder.get_instance()
    
    text_emb = clip.get_text_embedding(text)
    best_similarity = -1
    best_image = None
    
    for img_path in image_paths:
        img_emb = clip.get_image_embedding(img_path)
        if img_emb is not None:
            similarity = np.dot(text_emb, img_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_image = img_path
                
    return best_image, best_similarity
```

## Zarządzanie Zasobami

### CLIPEmbedder Resource Management
```python
# Automatyczne czyszczenie GPU cache
if self.device.startswith("cuda"):
    torch.cuda.empty_cache()

# Zamknięcie i reset singletona
clip.close()  # Zwolnienie modelu i pamięci GPU
```

### CohereEmbedder Key Rotation
```python
# Automatyczne wykrywanie błędów quota
error_keywords = ['quota', 'limit', 'trial', 'billing', 'exceeded', 
                  'usage limit', 'payment required', 'insufficient credits']

# Rotacja do następnego klucza
if any(keyword in error_msg for keyword in error_keywords):
    self._switch_to_next_key()
```

## Konfiguracja i Ścieżki

### Struktura Projektu
```
pgverse/
├── rag_codes/
│   ├── rag_functions/
│   │   └── embeddings.py
│   └── settings/
│       └── api_keys.json
└── data/
    └── images/
```

### Automatyczne Ścieżki
- Base path: automatyczne wykrywanie folderu głównego
- Image paths: inteligentne rozwiązywanie względem pgverse
- API keys: ścieżka względem struktury projektu

## Wymagania

### Python Packages
```
cohere>=4.0
torch>=1.9.0
transformers>=4.15.0
pillow>=8.0.0
scikit-learn>=1.0.0
numpy>=1.21.0
```

### Hardware
- **CPU**: Dowolny współczesny procesor
- **GPU**: CUDA-compatible (opcjonalnie, znacząco przyspiesza)
- **RAM**: Min. 4GB (8GB+ z GPU)
- **VRAM**: Min. 6GB dla pełnego modelu CLIP

## Optymalizacje

### GPU Utilization
- Automatyczne wykrywanie najlepszego GPU
- Batch processing dla większych zestawów
- Memory cleanup po operacjach
- CUDA cache management

### Memory Management  
- Lazy loading modeli
- Singleton pattern zapobiega duplikatom
- Automatyczne garbage collection
- Context managers dla operacji

### Error Handling
- Graceful degradation przy braku GPU
- Fallback strategies dla każdej operacji
- Comprehensive logging
- Resource cleanup w finally blocks