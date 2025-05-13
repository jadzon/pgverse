import os
import numpy as np
import cohere

# --- KONFIGURACJA ---
cohere_api_key = "xSywHzTHlEcq51tOI8rpxwWwtDdQnio5H7pPnuxs"
input_folder = "data/output/chunks"
embedding_folder = "embeddings"

os.makedirs(embedding_folder, exist_ok=True)

class TextEmbedder:
    def __init__(self, api_key: str, model_name: str = "embed-multilingual-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model_name
        
    def get_text_embedding(self, text: str, input_type: str = "search_document") -> np.ndarray:
        response = self.client.embed(
            texts=[text], 
            model=self.model,
            input_type=input_type
        )
        embedding = np.array(response.embeddings[0])
        return embedding

def process_and_save_embeddings(api_key: str):
    text_embedder = TextEmbedder(api_key)
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            embedding = text_embedder.get_text_embedding(text)
            # Zapis embeddingu do pliku .npy
            embedding_path = os.path.join(embedding_folder, filename.replace('.txt', '.npy'))
            np.save(embedding_path, embedding)
            print(f"Zapisano embedding dla {filename} do {embedding_path}")

if __name__ == "__main__":
    process_and_save_embeddings(cohere_api_key)
