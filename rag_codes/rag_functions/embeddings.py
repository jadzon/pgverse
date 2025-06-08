from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
import os
from pathlib import Path

class CLIPEmbedder:
    """
    Singleton - klasa do tworzenia embeddingów tekstu i obrazów przy użyciu modelu CLIP z Hugging Face
    Używa wzorca Singleton aby zapewnić tylko jedną instancję modelu w całej aplikacji
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, model_name="openai/clip-vit-base-patch32"):
        """Singleton - zawsze zwraca tę samą instancję"""
        if cls._instance is None:
            cls._instance = super(CLIPEmbedder, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Inicjalizuje embedder CLIP - TYLKO RAZ dzięki Singleton
        
        Args:
            model_name: Nazwa modelu CLIP z Hugging Face
        """
        # Inicjalizuj tylko raz
        if self._initialized:
            print("✅ CLIPEmbedder już zainicjalizowany - używam istniejącej instancji")
            return
            
        print(f"🔄 Pierwsza inicjalizacja CLIPEmbedder z modelem: {model_name}")
        
        # WYMUSZENIE CPU - całkowite wyłączenie CUDA
        self.device = "cpu"  # Zawsze CPU, ignoruj CUDA
        
        # Dodatkowe zabezpieczenia przed CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Ukryj wszystkie GPU
        torch.set_default_tensor_type('torch.FloatTensor')  # CPU tensors
        
        # Załaduj model i WYMUŚ CPU
        self.model = CLIPModel.from_pretrained(model_name)
        self.model = self.model.to("cpu")  # Explicite na CPU
        self.model.eval()  # Tryb ewaluacji
        
        # Wymuś CPU dla wszystkich parametrów modelu
        for param in self.model.parameters():
            param.data = param.data.cpu()
            if param.grad is not None:
                param.grad.data = param.grad.data.cpu()
        
        # Wymuś CPU dla wszystkich buforów modelu
        for buffer in self.model.buffers():
            buffer.data = buffer.data.cpu()
        
        # Załaduj procesor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        print(f"Model CLIP załadowany pomyślnie na {self.device} (CPU ONLY)")
        
        # Sprawdź wymiar embeddingów modelu - TYLKO CPU
        with torch.no_grad():
            test_inputs = self.processor(text=["test"], return_tensors="pt")
            # Upewnij się że wszystkie tensory są na CPU
            test_inputs = {k: v.cpu() if torch.is_tensor(v) else v for k, v in test_inputs.items()}
            
            test_embedding = self.model.get_text_features(**test_inputs)
            test_embedding = test_embedding.cpu()  # Wymuś CPU dla wyniku
            self.embedding_dim = test_embedding.shape[1]
        
        print(f"Wymiar embeddingów: {self.embedding_dim}")
        
        # Dodaj ścieżkę bazową do naprawiania ścieżek
        self.base_path = Path(__file__).parent.parent.parent.resolve()
        
        # Oznacz jako zainicjalizowany
        CLIPEmbedder._initialized = True
        print("✅ CLIPEmbedder zainicjalizowany w trybie CPU ONLY (SINGLETON)")
    
    @classmethod
    def get_instance(cls, model_name="openai/clip-vit-base-patch32"):
        """
        Metoda klasowa do pobierania instancji Singleton
        
        Args:
            model_name: Nazwa modelu (używana tylko przy pierwszej inicjalizacji)
            
        Returns:
            CLIPEmbedder: Singleton instancja
        """
        if cls._instance is None:
            cls._instance = cls(model_name)
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
        """Generuje embedding dla obrazu - TYLKO CPU"""
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
            
            # Przetwórz obraz TYLKO na CPU
            inputs = self.processor(images=image, return_tensors="pt")
            
            # WYMUSZENIE CPU dla wszystkich tensorów wejściowych
            inputs = {k: v.cpu() if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Upewnij się że model jest na CPU
            self.model = self.model.cpu()
            
            with torch.no_grad():
                # Generuj embedding na CPU
                image_features = self.model.get_image_features(**inputs)
                
                # Wymuś CPU dla wyniku
                image_features = image_features.cpu()
                
                # Normalizacja na CPU
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # Konwertuj do numpy (automatycznie CPU)
                result = image_features.squeeze().detach().numpy()
            
            return result
            
        except Exception as e:
            print(f"Błąd podczas generowania embeddingu obrazu {image_path}: {e}")
            return None

    def get_text_embedding(self, text):
        """Generuje embedding dla tekstu - TYLKO CPU"""
        try:
            if not text or not text.strip():
                return None
            
            # Upewnij się że model jest na CPU
            self.model = self.model.cpu()
            
            with torch.no_grad():
                # Przetwórz tekst na CPU
                inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
                
                # WYMUSZENIE CPU dla wszystkich tensorów wejściowych
                inputs = {k: v.cpu() if torch.is_tensor(v) else v for k, v in inputs.items()}
                
                # Generuj embedding na CPU
                embedding = self.model.get_text_features(**inputs)
                
                # Wymuś CPU dla wyniku
                embedding = embedding.cpu()
                
                # Konwertuj do numpy i znormalizuj (na CPU)
                embedding = embedding.detach().numpy()[0]
                embedding = normalize([embedding], norm='l2')[0]
            
            return embedding
            
        except Exception as e:
            print(f"Błąd podczas generowania embeddingu tekstu: {e}")
            return None
    
    def close(self):
        """Opcjonalne zamknięcie i zwolnienie zasobów"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            
            # Wyczyść cache PyTorch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reset singleton
            CLIPEmbedder._instance = None
            CLIPEmbedder._initialized = False
            
            print("✅ CLIPEmbedder zamknięty i zasoby zwolnione")
        except Exception as e:
            print(f"⚠ Błąd podczas zamykania CLIPEmbedder: {e}")