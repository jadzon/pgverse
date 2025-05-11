import numpy as np
import warnings
import cohere
import base64


# Suppress symlink warnings
warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")

class ImageEmbedder:
    def __init__(self, api_key: str, model_name: str = "embed-multilingual-v3.0"):
        """
        Initializes Cohere client for calculating image embeddings.
        
        Args:
            api_key: Cohere API key
            model_name: Model name to use (default: embed-multilingual-v3.0)
        """
        self.client = cohere.Client(api_key)
        self.model = model_name
        
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Calculates embedding for a single image using Cohere's API.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding vector as numpy array
        """
        # Read the image and convert to base64
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
            
        # Convert image to base64 string
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Get embedding from Cohere
        response = self.client.embed(
            texts=[""],  # Empty text
            model=self.model,
            input_type="image",
            image=base64_image
        )
        
        # Extract the embedding
        embedding = np.array(response.embeddings[0])
        
        # Normalize the embedding vector (as was done in the original implementation)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding

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