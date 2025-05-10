import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import warnings
import cohere

# Suppress symlink warnings
warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")

class ImageEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initializes CLIP model for calculating image embeddings.
        
        Args:
            model_name: Model name to load (default: CLIP from OpenAI)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Calculates embedding for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized embedding vector as numpy array
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize embedding vector
        image_embedding = image_features.cpu().numpy()[0]
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        
        return image_embedding

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