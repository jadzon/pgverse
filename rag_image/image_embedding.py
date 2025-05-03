import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import os
from typing import List, Tuple, Union, Optional
import matplotlib.pyplot as plt

class ImageEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Inicjalizuje model CLIP do obliczania embeddingów obrazów.
        
        Args:
            model_name: Nazwa modelu do załadowania (domyślnie CLIP od OpenAI)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Używanie urządzenia: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Oblicza embedding dla pojedynczego obrazu.
        
        Args:
            image_path: Ścieżka do pliku obrazu
            
        Returns:
            Wektor embeddingu jako tablica numpy
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalizacja wektora embeddingu
        image_embedding = image_features.cpu().numpy()[0]
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        
        return image_embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Oblicza podobieństwo kosynusowe między dwoma embeddingami.
        
        Args:
            embedding1: Pierwszy wektor embeddingu
            embedding2: Drugi wektor embeddingu
            
        Returns:
            Wartość podobieństwa kosynusowego (od -1 do 1, gdzie 1 oznacza identyczne)
        """
        return np.dot(embedding1, embedding2)
    
    def compare_images(self, image_path1: str, image_path2: str) -> float:
        """
        Porównuje dwa obrazy i zwraca ich podobieństwo.
        
        Args:
            image_path1: Ścieżka do pierwszego obrazu
            image_path2: Ścieżka do drugiego obrazu
            
        Returns:
            Wartość podobieństwa kosynusowego między obrazami
        """
        embedding1 = self.get_image_embedding(image_path1)
        embedding2 = self.get_image_embedding(image_path2)
        
        similarity = self.compute_similarity(embedding1, embedding2)
        return similarity

def visualize_comparison(image_path1: str, image_path2: str, similarity: float):
    """
    Wizualizuje porównanie dwóch obrazów wraz z ich wartością podobieństwa.
    
    Args:
        image_path1: Ścieżka do pierwszego obrazu
        image_path2: Ścieżka do drugiego obrazu
        similarity: Wartość podobieństwa między obrazami
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')
    
    ax1.imshow(img1)
    ax1.set_title("Obraz 1")
    ax1.axis('off')
    
    ax2.imshow(img2)
    ax2.set_title("Obraz 2")
    ax2.axis('off')
    
    plt.suptitle(f"Podobieństwo kosynusowe: {similarity:.4f}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Przykładowe użycie
    try:
        # Sprawdzamy, czy mamy zainstalowane potrzebne biblioteki
        import transformers
    except ImportError:
        print("Instaluję wymagane biblioteki...")
        os.system("pip install transformers torch Pillow matplotlib numpy")
        print("Biblioteki zainstalowane, uruchom skrypt ponownie.")
        exit()
    
    # Definiujemy ścieżki do obrazów w folderze images
    images_dir = "image"
    
    # Sprawdzamy czy folder images istnieje, jeśli nie - tworzymy go
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Utworzono folder {images_dir}. Umieść tam obrazy o nazwach image1.* i image2.* (np. PNG, JPG).")
        exit(1)
    
    # Szukamy plików image1 i image2 z dowolnym rozszerzeniem
    image1_files = [f for f in os.listdir(images_dir) if f.startswith('image1.')]
    image2_files = [f for f in os.listdir(images_dir) if f.startswith('image2.')]
    
    if not image1_files or not image2_files:
        print("Nie znaleziono plików image1.* lub image2.* w folderze images.")
        print("Upewnij się, że pliki znajdują się w folderze i mają odpowiednie nazwy.")
        exit(1)
    
    # Używamy pierwszego znalezionego pliku dla każdego obrazu
    image_path1 = os.path.join(images_dir, image1_files[0])
    image_path2 = os.path.join(images_dir, image2_files[0])
    
    print(f"Znaleziono obrazy: {image_path1} i {image_path2}")
    
    # Inicjalizacja embeddera i obliczenie podobieństwa
    print("Ładowanie modelu CLIP...")
    embedder = ImageEmbedder()
    
    print("Obliczanie podobieństwa obrazów...")
    similarity = embedder.compare_images(image_path1, image_path2)
    
    print(f"Podobieństwo kosynusowe między obrazami: {similarity:.4f}")
    
    # Wizualizacja wyników
    visualize_comparison(image_path1, image_path2, similarity)