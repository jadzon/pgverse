from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize

class CLIPEmbedder:
    """
    Klasa do tworzenia embeddingów tekstu i obrazów przy użyciu modelu CLIP z Hugging Face
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Inicjalizuje embedder CLIP
        
        Args:
            model_name: Nazwa modelu CLIP z Hugging Face
        """
        
        print(f"Ładowanie modelu CLIP: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"Model CLIP załadowany pomyślnie na {self.device}")
        
        # Sprawdź wymiar embeddingów modelu
        with torch.no_grad():
            test_inputs = self.processor(text=["test"], return_tensors="pt").to(self.device)
            test_embedding = self.model.get_text_features(**test_inputs)
            self.embedding_dim = test_embedding.shape[1]
        print(f"Wymiar embeddingów: {self.embedding_dim}")
        
    def get_text_embedding(self, text):
        """Generuje znormalizowany embedding tekstu używając CLIP"""
        
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            embedding = self.model.get_text_features(**inputs)
            
            # Konwertuj do numpy i znormalizuj
            embedding = embedding.cpu().numpy()[0]
            embedding = normalize([embedding], norm='l2')[0]
            
        return embedding
    
    def get_image_embedding(self, image_path):
        """Generuje znormalizowany embedding obrazu używając CLIP"""
        
        try:
            # Wczytaj obraz
            image = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                # Użyj tylko parametru images, bez tekstu
                inputs = self.processor(images=image, return_tensors="pt")
                
                # Przenieś tylko tensory obrazów na urządzenie
                pixel_values = inputs['pixel_values'].to(self.device)
                
                # Generuj embedding używając bezpośrednio pixel_values
                embedding = self.model.get_image_features(pixel_values=pixel_values)
                
                # Konwertuj do numpy i znormalizuj
                embedding = embedding.cpu().numpy()[0]
                embedding = normalize([embedding], norm='l2')[0]
                
            return embedding
            
        except Exception as e:
            print(f"Błąd podczas generowania embeddingu obrazu {image_path}: {e}")
            # Zwróć znormalizowany wektor zerowy o odpowiednim wymiarze
            zero_vector = np.zeros(self.embedding_dim)
            return normalize([zero_vector], norm='l2')[0]