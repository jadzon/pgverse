import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cohere
import os

class CohereTextChunker:
    def __init__(self, api_key, model_name='embed-multilingual-v3.0'):
        """
        Inicjalizuje chunker tekstu z określonym modelem Cohere do embeddingów.
        
        Args:
            api_key (str): Klucz API dla Cohere
            model_name (str): Nazwa modelu Cohere do użycia (domyślnie: embed-multilingual-v3.0)
        """
        self.client = cohere.Client(api_key)
        self.model = model_name
    
    def chunk_from_file(self, file_path, max_tokens=500):
        """
        Dzieli dokument z pliku na semantyczne chunki.
        
        Args:
            file_path (str): Ścieżka do pliku tekstowego
            max_tokens (int): Maksymalna liczba tokenów w jednym chunku
            
        Returns:
            list: Lista chunków tekstowych
        """
        # Sprawdź czy plik istnieje
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Plik {file_path} nie istnieje")
        
        # Wczytaj zawartość pliku
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Użyj metody chunk_text do przetworzenia tekstu
        return self.chunk_text(text, max_tokens)
    
    def chunk_text(self, text, max_tokens=500):
        """
        Dzieli podany tekst na semantyczne chunki.
        
        Args:
            text (str): Tekst do podziału
            max_tokens (int): Maksymalna liczba tokenów w jednym chunku
            
        Returns:
            list: Lista chunków tekstowych
        """
        # Podziel dokument na rozdziały (jeśli istnieją)
        chapters = re.split(r'(ROZDZIAŁ|STAVE|CHAPTER) [IVXLCDM\d]+\.?', text)
        
        # Jeśli nie znaleziono rozdziałów, traktuj cały tekst jako jeden rozdział
        if len(chapters) <= 1:
            chapters = [text]
        else:
            # Połącz nagłówki z treścią rozdziałów
            processed_chapters = []
            for i in range(0, len(chapters), 2):
                chapter_title = chapters[i] if i == 0 else chapters[i-1] + chapters[i]
                chapter_content = chapters[i+1] if i+1 < len(chapters) else ""
                
                if not chapter_content.strip():
                    continue
                    
                # Dodaj tytuł rozdziału jako kontekst
                if chapter_title.strip():
                    processed_chapters.append(f"{chapter_title.strip()}\n\n{chapter_content.strip()}")
                else:
                    processed_chapters.append(chapter_content.strip())
            
            chapters = processed_chapters
        
        # Przetwórz każdy rozdział na chunki
        all_chunks = []
        for chapter in chapters:
            if not chapter.strip():
                continue
                
            # Wstępnie podziel na fragmenty bazowe (~15 tokenów)
            base_fragments = self._simple_recursive_split(chapter, target_tokens=15)
            n = len(base_fragments)
            
            if n == 0:
                continue
            
            # Oblicz embeddingi dla fragmentów używając Cohere
            embeddings = self._get_embeddings(base_fragments)
            
            # Oblicz macierz podobieństwa
            sim_matrix = cosine_similarity(embeddings)
            
            # Użyj programowania dynamicznego do znalezienia optymalnego podziału
            chunks = self._find_optimal_chunks(base_fragments, sim_matrix, max_tokens)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _get_embeddings(self, texts):
        """
        Generuje embeddingi dla listy tekstów używając Cohere API.
        
        Args:
            texts (list): Lista tekstów do wygenerowania embeddingów
            
        Returns:
            np.ndarray: Macierz embeddingów, gdzie każdy wiersz to embedding tekstu
        """
        # Generowanie embeddingów w pakietach po 96 tekstów (limit Cohere)
        batch_size = 96
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = self.client.embed(
                texts=batch_texts,
                model=self.model,
                input_type="search_document"
            )
            all_embeddings.extend(response.embeddings)
        
        return np.array(all_embeddings)

    def _simple_recursive_split(self, text, target_tokens):
        """
        Dzieli tekst na małe fragmenty bazowe.
        """
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for s in sentences:
            s_tokens = len(s.split())
            if current_tokens + s_tokens > target_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = s
                current_tokens = s_tokens
            else:
                current_chunk += " " + s if current_chunk else s
                current_tokens += s_tokens
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _find_optimal_chunks(self, fragments, sim_matrix, max_tokens):
        """
        Znajduje optymalny podział fragmentów na chunki używając programowania dynamicznego.
        """
        n = len(fragments)
        # Oblicz liczbę tokenów dla każdego fragmentu
        token_counts = [len(fragment.split()) for fragment in fragments]
        
        # dp[i] = maksymalny reward dla segmentacji fragmentów i..n-1
        dp = np.full(n+1, -np.inf)
        dp[n] = 0  # base case
        next_split = [-1] * (n + 1)
        
        # Funkcja reward dla chunka
        def chunk_reward(i, j):
            if j - i < 2:
                return 0
            reward = 0
            for p in range(i, j):
                for q in range(p+1, j):
                    reward += sim_matrix[p, q]
            return reward
        
        # Algorytm programowania dynamicznego
        for i in range(n-1, -1, -1):
            total_tokens = 0
            best_reward = -np.inf
            best_j = i + 1
            
            for j in range(i+1, n+1):
                if j > i:
                    total_tokens += token_counts[j-1]
                if total_tokens > max_tokens:
                    break
                    
                current_reward = chunk_reward(i, j) + dp[j]
                if current_reward > best_reward:
                    best_reward = current_reward
                    best_j = j
                    
            dp[i] = best_reward
            next_split[i] = best_j
        
        # Odtwórz optymalny podział
        chunks = []
        i = 0
        while i < n:
            j = next_split[i]
            chunk_text = " ".join(fragments[i:j])
            chunks.append(chunk_text)
            i = j
        
        return chunks