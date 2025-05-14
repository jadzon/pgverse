import numpy as np
import warnings
import cohere
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
# Suppress symlink warnings
warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")

class ImageEmbedder:
    def __init__(self, api_key=None):
        """
        Initializes the CLIP model for image embeddings.
        The api_key parameter is kept for backward compatibility but not used.
        """
        print("Loading CLIP multilingual model...")
        self.model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
        print("CLIP model loaded successfully")
    
    def get_image_embedding(self, image_path):
        """
        Creates an embedding for an image using CLIP.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            numpy.ndarray: The embedding vector
        """
        try:
            # Process and get embedding directly from the image path
            # SentenceTransformer's CLIP can process image paths directly
            embedding = self.model.encode(image_path, show_progress_bar=False)
            return np.array(embedding)
            
        except Exception as e:
            print(f"Error creating image embedding for {image_path}: {e}")
            raise

class TextEmbedder:
    def __init__(self, api_key: str, model_name: str = "embed-multilingual-v3.0"):
        """
        Initializes Cohere client for calculating text embeddings.
        
        Args:
            api_key: Cohere API key
            model_name: Model name to use (default: embed-multilingual-v3.0)
        """
        self.client = cohere.Client(api_key)
        self.model = model_name
        
    def get_text_embedding(self, text: str, input_type: str = "search_document") -> np.ndarray:
        """
        Calculates embedding for a single text.
        
        Args:
            text: Text to embed
            input_type: Type of input ("search_document" or "search_query")
            
        Returns:
            Embedding vector as numpy array
        """
        response = self.client.embed(
            texts=[text], 
            model=self.model,
            input_type=input_type
        )
        
        # Extract and return the embedding
        embedding = np.array(response.embeddings[0])
        
        return embedding
    # Dodaj nową klasę CLIPEmbedder
class CLIPEmbedder:
    """
    Klasa do tworzenia embeddingów tekstu i obrazów przy użyciu wielojęzycznego modelu CLIP
    z obsługą języka polskiego
    """
    def __init__(self, model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1"):
        """
        Inicjalizuje wielojęzyczny embedder CLIP z obsługą języka polskiego
        
        Args:
            model_name: Nazwa modelu CLIP z biblioteki sentence-transformers
        """
        
        print(f"Ładowanie wielojęzycznego modelu CLIP z obsługą języka polskiego: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Model CLIP załadowany pomyślnie")
        
    def get_text_embedding(self, text):
        """Generuje embedding tekstu używając wielojęzycznego CLIP"""
        import numpy as np
        
        # SentenceTransformer obsługuje bezpośrednio wielojęzyczne zapytania
        embedding = self.model.encode(text, show_progress_bar=False)
        
        # Normalizacja wektora
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized_emb = embedding / norm
        else:
            normalized_emb = embedding
            
        return normalized_emb
    
    def get_image_embedding(self, image_path):
        """Generuje embedding obrazu używając wielojęzycznego CLIP"""
        import numpy as np
        
        try:
            # SentenceTransformer może bezpośrednio przetwarzać ścieżki do obrazów
            embedding = self.model.encode(image_path, show_progress_bar=False)
            
            # Normalizacja wektora
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized_emb = embedding / norm
            else:
                normalized_emb = embedding
                
            return normalized_emb
            
        except Exception as e:
            print(f"Błąd podczas generowania embeddingu obrazu {image_path}: {e}")
            # Zwróć wektor zerowy w przypadku błędu
            return np.zeros(512)  # Domyślnie 512-wymiarowy wektor dla CLIP