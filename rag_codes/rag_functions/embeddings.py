from pathlib import Path
import cohere
import json
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import warnings
import os
from sklearn.preprocessing import normalize
# Wycisz warningi
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class CohereEmbedder:
    """
    Uproszczony Singleton - klasa do tworzenia embeddingów tekstu przy użyciu Cohere API
    Automatyczne przełączanie kluczy przy błędach
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, api_key=None):
        """Singleton - zawsze zwraca tę samą instancję"""
        if cls._instance is None:
            cls._instance = super(CohereEmbedder, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, api_key=None):
        """Inicjalizuje Cohere embedder - TYLKO RAZ dzięki Singleton"""
        if self._initialized:
            return
            
        print("🔄 Inicjalizacja CohereEmbedder")
        
        # Ścieżka do pliku z kluczami
        self.base_path = Path(__file__).parent.parent.parent.resolve()
        self.api_keys_file = self.base_path / "rag_codes" / "settings" / "api_keys.json"
        
        # Wczytaj klucze API
        self._available_keys = []
        self._current_key_index = 0
        self._load_api_keys()
        
        # Ustaw aktualny klucz
        self._api_key = api_key or (self._available_keys[0] if self._available_keys else None)
        self._client = None
        
        # Oznacz jako zainicjalizowany
        CohereEmbedder._initialized = True
        print("✅ CohereEmbedder zainicjalizowany")
    
    def _load_api_keys(self):
        """Wczytuje klucze API z pliku api_keys.json"""
        try:
            if self.api_keys_file.exists():
                with open(self.api_keys_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        self._available_keys = data
                    elif isinstance(data, dict) and 'cohere_keys' in data:
                        self._available_keys = data['cohere_keys']
                    else:
                        self._available_keys = []
                
                print(f"🔑 Wczytano {len(self._available_keys)} kluczy API")
            else:
                print(f"⚠️ Plik {self.api_keys_file} nie istnieje")
                self._available_keys = []
        
        except Exception as e:
            print(f"❌ Błąd wczytywania kluczy API: {e}")
            self._available_keys = []
    
    def _switch_to_next_key(self):
        """Przełącza na następny dostępny klucz API"""
        if not self._available_keys:
            return False
        
        self._current_key_index = (self._current_key_index + 1) % len(self._available_keys)
        old_key = self._api_key
        self._api_key = self._available_keys[self._current_key_index]
        
        print(f"🔄 Przełączanie klucza:")
        print(f"   Z: {old_key[:8] if old_key else 'None'}...{old_key[-4:] if old_key else 'None'}")
        print(f"   Na: {self._api_key[:8]}...{self._api_key[-4:]}")
        
        # Wymuś nowy client
        self._client = None
        return True
    
    def _ensure_client_initialized(self):
        """Tworzy client jeśli nie istnieje"""
        if self._client is None:
            if not self._api_key:
                raise ValueError("❌ Brak klucza API Cohere!")
            
            self._client = cohere.Client(api_key=self._api_key)
            print(f"✅ Klient Cohere utworzony dla klucza: {self._api_key[:8]}...{self._api_key[-4:]}")

    def get_text_embedding(self, text, model="embed-v4.0", input_type="search_document"):
        """
        Generuje embedding dla tekstu z automatycznym przełączaniem kluczy
        
        Args:
            text: Tekst do embeddingu
            model: Model Cohere
            input_type: Typ wejścia
            
        Returns:
            numpy.ndarray: Embedding tekstu lub None w przypadku błędu
        """
        if not text or not text.strip():
            print("⚠️ Pusty tekst")
            return None
        
        # Skróć tekst jeśli za długi
        if len(text) > 8000:
            text = text[:8000]
            print(f"⚠️ Tekst skrócony do {len(text)} znaków")
        
        # Próbuj z każdym dostępnym kluczem
        keys_tried = 0
        max_keys = len(self._available_keys) if self._available_keys else 1
        
        while keys_tried < max_keys:
            try:
                self._ensure_client_initialized()
                
                print(f"🔄 Generowanie embeddingu... (klucz {keys_tried + 1}/{max_keys})")
                
                # Wywołanie API Cohere
                response = self._client.embed(
                    texts=[text],
                    model=model,
                    input_type=input_type,
                    truncate="END"
                )
                
                # Sprawdź czy embedding jest poprawny
                if response and response.embeddings and len(response.embeddings) > 0:
                    embedding = np.array(response.embeddings[0], dtype=np.float32)
                    
                    # Sprawdź czy embedding nie jest pusty
                    if embedding is not None and len(embedding) > 0:
                        print(f"✅ Embedding wygenerowany (wymiar: {len(embedding)})")
                        return embedding
                    else:
                        print("⚠️ Embedding jest pusty")
                else:
                    print("⚠️ Brak odpowiedzi z API")
                
                # Jeśli embedding jest pusty, przejdź do następnego klucza
                keys_tried += 1
                if keys_tried < max_keys:
                    if self._switch_to_next_key():
                        continue
                    else:
                        break
                        
            except Exception as e:
                print(f"❌ Błąd API (klucz {keys_tried + 1}): {e}")
                
                # Sprawdź czy to błąd limitu quota
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in [
                    'quota', 'limit', 'trial', 'billing', 'exceeded',
                    'usage limit', 'payment required', 'insufficient credits'
                ]):
                    print("🚨 Wykryto limit quota - przełączam klucz")
                
                keys_tried += 1
                if keys_tried < max_keys:
                    if self._switch_to_next_key():
                        continue
                    else:
                        break
                else:
                    break
        
        # Jeśli wszystkie klucze nie działają
        print("⚠️ WARNING: Nie udało się wygenerować embeddingu żadnym z dostępnych kluczy!")
        return None

    def get_text_embeddings_batch(self, texts, model="embed-v4.0", input_type="search_document", batch_size=96):
        """
        Generuje embeddingi dla wielu tekstów w batch'ach
        
        Args:
            texts: Lista tekstów
            model: Model Cohere
            input_type: Typ wejścia
            batch_size: Rozmiar batcha (max 96)
            
        Returns:
            list: Lista embeddingów numpy.ndarray lub None
        """
        if not texts:
            return []
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"🔄 Przetwarzanie {len(texts)} tekstów w {total_batches} batch'ach")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"📦 Batch {batch_num}/{total_batches}: {len(batch)} tekstów")
            
            # Filtruj puste teksty
            valid_texts = [text for text in batch if text and text.strip()]
            if not valid_texts:
                embeddings.extend([None] * len(batch))
                continue
            
            # Skróć długie teksty
            processed_texts = []
            for text in valid_texts:
                if len(text) > 8000:
                    text = text[:8000]
                processed_texts.append(text)
            
            # Próbuj z każdym kluczem
            keys_tried = 0
            max_keys = len(self._available_keys) if self._available_keys else 1
            batch_success = False
            
            while keys_tried < max_keys and not batch_success:
                try:
                    self._ensure_client_initialized()
                    
                    # API call
                    response = self._client.embed(
                        texts=processed_texts,
                        model=model,
                        input_type=input_type,
                        truncate="END"
                    )
                    
                    if response and response.embeddings:
                        batch_embeddings = [np.array(emb, dtype=np.float32) for emb in response.embeddings]
                        embeddings.extend(batch_embeddings)
                        print(f"✅ Batch {batch_num} zakończony pomyślnie")
                        batch_success = True
                    else:
                        print(f"⚠️ Batch {batch_num}: Brak odpowiedzi z API")
                        
                except Exception as batch_error:
                    print(f"❌ Błąd batch {batch_num} (klucz {keys_tried + 1}): {batch_error}")
                    
                keys_tried += 1
                if keys_tried < max_keys and not batch_success:
                    if not self._switch_to_next_key():
                        break
            
            if not batch_success:
                embeddings.extend([None] * len(valid_texts))
                print(f"⚠️ WARNING: Batch {batch_num} nieudany - wszystkie klucze wyczerpane")
        
        success_count = sum(1 for emb in embeddings if emb is not None)
        print(f"📊 Pomyślnie wygenerowano {success_count}/{len(texts)} embeddingów")
        
        return embeddings
    
    @classmethod
    def get_instance(cls, api_key=None):
        """Pobiera instancję Singleton"""
        if cls._instance is None:
            cls._instance = cls(api_key)
        return cls._instance
    
    def close(self):
        """Zamyka i resetuje embedder"""
        try:
            self._client = None
            CohereEmbedder._instance = None
            CohereEmbedder._initialized = False
            print("✅ CohereEmbedder zamknięty")
        except Exception as e:
            print(f"⚠️ Błąd zamykania: {e}")


class CLIPEmbedder:
    """
    Singleton - klasa do tworzenia embeddingów tekstu i obrazów przy użyciu modelu openai/clip-vit-large-patch14
    Automatyczne wykrywanie CUDA i użycie GPU gdy dostępne
    Konwersja z 768 wymiarów na 1024 wymiary
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton - zawsze zwraca tę samą instancję"""
        if cls._instance is None:
            cls._instance = super(CLIPEmbedder, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Inicjalizuje embedder CLIP - TYLKO RAZ dzięki Singleton
        Model: openai/clip-vit-large-patch14 (768 wymiarów -> 1024 wymiary)
        """
        # Inicjalizuj tylko raz
        if self._initialized:
            print("✅ CLIPEmbedder już zainicjalizowany - używam istniejącej instancji")
            return
        
        # AUTOMATYCZNE WYKRYWANIE CUDA
        self._detect_device()
        
        # Załaduj model
        try:
            print("📥 Ładowanie modelu openai/clip-vit-large-patch14...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.model_name = "openai/clip-vit-large-patch14"
            
            # Przenieś model na wykryte urządzenie
            print(f"🚀 Przenoszenie modelu na {self.device}...")
            self.model = self.model.to(self.device)
            self.model.eval()  # Tryb ewaluacji
            
            print(f"✅ Model openai/clip-vit-large-patch14 załadowany pomyślnie na {self.device}")
            
        except Exception as e:
            print(f"❌ Błąd ładowania modelu: {e}")
            raise RuntimeError(f"Nie udało się załadować modelu openai/clip-vit-large-patch14: {e}")
        
        # Inicjalizuj warstwę konwersji wymiarów
        self._init_dimension_converter()
        
        # Dodaj ścieżkę bazową
        self.base_path = Path(__file__).parent.parent.parent.resolve()
        
        # Oznacz jako zainicjalizowany
        CLIPEmbedder._initialized = True
        print("✅ CLIPEmbedder zainicjalizowany (SINGLETON)")
    
    def _detect_device(self):
        """Automatyczne wykrywanie najlepszego dostępnego urządzenia"""
        if torch.cuda.is_available():
            # Sprawdź ilość pamięci GPU
            gpu_count = torch.cuda.device_count()
            print(f"🎮 Wykryto {gpu_count} GPU")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Wybierz główny GPU
            self.device = f"cuda:0"
            print(f"✅ Wybrano urządzenie: {self.device}")
            
            # Wyczyść cache GPU
            torch.cuda.empty_cache()
            
        else:
            self.device = "cpu"
            print("⚠️ CUDA niedostępna - używam CPU")

    
    def _init_dimension_converter(self):
        """Inicjalizuje warstwę konwersji z 768 na 1024 wymiary"""
        print("🔧 Inicjalizacja konwertera wymiarów 768 -> 1024...")
        
        # Warstwa liniowa do konwersji wymiarów
        self.dimension_converter = torch.nn.Linear(768, 1024, bias=False).to(self.device)
        
        # Inicjalizacja Xavier/Glorot
        torch.nn.init.xavier_uniform_(self.dimension_converter.weight)
        
        # Ustaw tryb ewaluacji
        self.dimension_converter.eval()
        
        # Docelowy wymiar
        self.embedding_dim = 1024
    
    def _convert_to_1024_dims(self, embedding_768):
        """
        Konwertuje embedding z 768 na 1024 wymiary
        
        Args:
            embedding_768: numpy array o wymiarze 768
            
        Returns:
            numpy array o wymiarze 1024
        """
        try:
            # Konwertuj do torch tensor
            if isinstance(embedding_768, np.ndarray):
                tensor_768 = torch.from_numpy(embedding_768).float().to(self.device)
            else:
                tensor_768 = embedding_768.float().to(self.device)
            
            # Jeśli tensor ma więcej wymiarów, dodaj batch dimension
            if len(tensor_768.shape) == 1:
                tensor_768 = tensor_768.unsqueeze(0)
            
            with torch.no_grad():
                # Konwersja przez warstwę liniową
                tensor_1024 = self.dimension_converter(tensor_768)
                
                # Normalizacja L2
                tensor_1024 = tensor_1024 / tensor_1024.norm(p=2, dim=-1, keepdim=True)
                
                # Konwertuj z powrotem do numpy
                result = tensor_1024.squeeze().detach().cpu().numpy()
            
            return result
            
        except Exception as e:
            print(f"❌ Błąd konwersji wymiarów: {e}")
            # Fallback - padding z zerami
            if isinstance(embedding_768, np.ndarray):
                padding = np.zeros(1024 - len(embedding_768), dtype=embedding_768.dtype)
                result = np.concatenate([embedding_768, padding])
                # Normalizuj
                norm = np.linalg.norm(result)
                if norm > 0:
                    result = result / norm
                return result
            return None
    
    @classmethod
    def get_instance(cls):
        """
        Metoda klasowa do pobierania instancji Singleton
        
        Returns:
            CLIPEmbedder: Singleton instancja z modelem openai/clip-vit-large-patch14 (1024 wymiary)
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def is_initialized(cls):
        """Sprawdza czy Singleton został już zainicjalizowany"""
        return cls._initialized
    
    def fix_image_path(self, image_path):
        """
        Poprawia ścieżkę obrazu aby działała na każdym komputerze
        """
        try:
            # Konwertuj na Path object
            path_obj = Path(image_path)
            
            # Jeśli ścieżka jest już absolutna i istnieje
            if path_obj.is_absolute() and path_obj.exists():
                return str(path_obj)
            
            # Znajdź bazową ścieżkę projektu (folder zawierający DataBaseApp.py)
            current_file = Path(__file__).parent.parent.parent  # wyjdź z rag_functions/embeddings do głównego folderu
            
            # Jeśli ścieżka zaczyna się od "pgverse", usuń ten prefiks
            path_str = str(path_obj)
            if path_str.startswith("pgverse"):
                # Usuń "pgverse" i ewentualny separator na początku
                path_str = path_str[7:]  # "pgverse" ma 7 znaków
                path_str = path_str.lstrip(r'\/\/')  # usuń początkowe separatory
                path_obj = Path(path_str)
            
            # Zbuduj pełną ścieżkę względem głównego folderu projektu
            full_path = current_file / path_obj
            
            if full_path.exists():
                return str(full_path.resolve())
            
            # Jeśli nie istnieje, spróbuj różnych wariantów
            # Sprawdź czy parts zawierają "rag_codes"
            parts = path_obj.parts
            if "rag_codes" in parts:
                # Znajdź indeks "rag_codes" i zbuduj ścieżkę od tego miejsca
                rag_codes_index = parts.index("rag_codes")
                relative_path = Path(*parts[rag_codes_index:])
                full_path = current_file / relative_path
                
                if full_path.exists():
                    return str(full_path.resolve())
            
            # Ostatnia próba - załóż że ścieżka jest względna względem głównego folderu
            fallback_path = current_file / path_obj
            if fallback_path.exists():
                return str(fallback_path.resolve())
                
            return None
            
        except Exception as e:
            print(f"Błąd w fix_image_path: {e}")
            return None

    def get_image_embedding(self, image_path):
        """Generuje embedding dla obrazu - z automatycznym użyciem CUDA/CPU (1024 wymiary)"""
        try:
            # Napraw ścieżkę przed użyciem
            fixed_path = self.fix_image_path(image_path)
            if fixed_path is None:
                print(f"Błąd podczas naprawiania ścieżki obrazu {image_path}")
                return None
            
            # Sprawdź czy plik istnieje
            if not os.path.exists(fixed_path):
                print(f"Błąd: Plik obrazu nie istnieje: {fixed_path}")
                return None
            
            # Otwórz i przetwórz obraz
            image = Image.open(fixed_path)
            
            # Przetwórz obraz
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Przenieś tensory na właściwe urządzenie
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                # Generuj embedding (768 wymiarów)
                image_features = self.model.get_image_features(**inputs)
                
                # Normalizacja
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # Konwertuj do numpy (przenieś na CPU jeśli potrzeba)
                embedding_768 = image_features.squeeze().detach().cpu().numpy()
            
            # Konwertuj z 768 na 1024 wymiary
            embedding_1024 = self._convert_to_1024_dims(embedding_768)
            
            return embedding_1024
            
        except Exception as e:
            print(f"Błąd podczas generowania embeddingu obrazu {image_path}: {e}")
            return None

    def get_text_embedding(self, text):
        """Generuje embedding dla tekstu - z automatycznym użyciem CUDA/CPU (1024 wymiary)"""
        try:
            if not text or not text.strip():
                return None
            
            with torch.no_grad():
                # Przetwórz tekst
                inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
                
                # Przenieś tensory na właściwe urządzenie
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                
                # Generuj embedding (768 wymiarów)
                embedding = self.model.get_text_features(**inputs)
                
                # Konwertuj do numpy (przenieś na CPU) i znormalizuj
                embedding_768 = embedding.detach().cpu().numpy()[0]
                embedding_768 = normalize([embedding_768], norm='l2')[0]
            
            # Konwertuj z 768 na 1024 wymiary
            embedding_1024 = self._convert_to_1024_dims(embedding_768)
            
            return embedding_768
            
        except Exception as e:
            print(f"Błąd podczas generowania embeddingu tekstu: {e}")
            return None
    
    def close(self):
        """Zamknięcie i zwolnienie zasobów"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            if hasattr(self, 'dimension_converter'):
                del self.dimension_converter
            
            # Wyczyść cache GPU jeśli używane
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                print("🧹 Wyczyszczono cache GPU")
            
            # Reset singleton
            CLIPEmbedder._instance = None
            CLIPEmbedder._initialized = False
            
            print("✅ CLIPEmbedder zamknięty i zasoby zwolnione")
        except Exception as e:
            print(f"⚠️ Błąd podczas zamykania CLIPEmbedder: {e}")