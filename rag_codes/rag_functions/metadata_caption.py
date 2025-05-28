import re
import os
from chunker import TextChunker
from embeddings import CLIPEmbedder
import json
from sklearn.metrics.pairwise import cosine_similarity
class ImageTextProcessor:
    def __init__(self, chunker=None, max_tokens=500):
        """
        Inicjalizuje procesor do przetwarzania plików txt z obrazkami.
        
        Args:
            chunker: Instancja TextChunker (jeśli None, zostanie utworzona nowa)
            max_tokens (int): Maksymalna liczba tokenów w jednym chunku
        """
        if chunker is None:
            self.chunker = TextChunker()
        else:
            self.chunker = chunker
        self.max_tokens = max_tokens
    
    def process_file(self, file_path):
        """
        Przetwarza plik txt z obrazkami zgodnie z algorytmem:
        1. Wykryj obrazek
        2. Pochunkuj tekst przed obrazkiem
        3. Dodaj chunki do tablicy
        4. Dodaj ścieżkę obrazka do tablicy
        5. Powtarzaj aż do końca pliku
        
        Args:
            file_path (str): Ścieżka do pliku txt
            
        Returns:
            list: Tablica z chunkami tekstu i ścieżkami obrazków w odpowiedniej kolejności
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Plik {file_path} nie istnieje")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return self.process_text(content)
    
    def process_text(self, content):
        """
        Przetwarza tekst z obrazkami zgodnie z algorytmem.
        
        Args:
            content (str): Tekst do przetworzenia
            
        Returns:
            list: Tablica z chunkami tekstu i ścieżkami obrazków w odpowiedniej kolejności
        """
        texts = []
        current_position = 0
        
        image_pattern = r'<image/([^>]+)>'
        
        while current_position < len(content):
            match = re.search(image_pattern, content[current_position:])
            
            if match:
                image_start = current_position + match.start()
                image_end = current_position + match.end()
                image_path = match.group(1)
                
                text_before_image = content[current_position:image_start].strip()
                
                if text_before_image:
                    chunks = self.chunker.chunk_text(text_before_image, self.max_tokens)
                    texts.extend(chunks)
                
                texts.append(f"<image/{image_path}>")
                current_position = image_end
            
            else:
                remaining_text = content[current_position:].strip()
                if remaining_text:
                    chunks = self.chunker.chunk_text(remaining_text, self.max_tokens)
                    texts.extend(chunks)
                break
        
        return texts
    
    def get_element_type(self, element):
        """
        Określa typ elementu w tablicy.
        
        Args:
            element (str): Element z tablicy texts
            
        Returns:
            str: 'image' lub 'text'
        """
        if element.startswith('<image/') and element.endswith('>'):
            return 'image'
        else:
            return 'text'
    
    def get_image_path(self, image_element):
        """
        Wyciąga ścieżkę z elementu obrazka.
        
        Args:
            image_element (str): Element obrazka w formacie <image/path>
            
        Returns:
            str: Ścieżka do obrazka lub None jeśli element nie jest obrazkiem
        """
        if self.get_element_type(image_element) == 'image':
            return image_element[7:-1]
        return None
    
    def get_images_with_context_json(self, texts):
        """
        Zwraca JSON z obrazkami i ich sąsiednimi chunkami.
        
        Args:
            texts (list): Tablica z wynikami process_file/process_text
            
        Returns:
            list: Lista słowników w formacie [{"path": ["chunk górny", "chunk dolny"]}]
        """
        result = []
        
        for i, element in enumerate(texts):
            if self.get_element_type(element) == 'image':
                image_path = self.get_image_path(element)
                
                # Znajdź chunk górny (bezpośrednio przed obrazkiem)
                chunk_gorny = None
                for j in range(i - 1, -1, -1):
                    if self.get_element_type(texts[j]) == 'text':
                        chunk_gorny = texts[j]
                        break
                    elif self.get_element_type(texts[j]) == 'image':
                        break
                
                # Znajdź chunk dolny (bezpośrednio po obrazku)
                chunk_dolny = None
                for j in range(i + 1, len(texts)):
                    if self.get_element_type(texts[j]) == 'text':
                        chunk_dolny = texts[j]
                        break
                    elif self.get_element_type(texts[j]) == 'image':
                        break
                
                # Twórz wpis zgodnie z wymaganiami
                chunks = []
                if chunk_gorny is not None:
                    chunks.append(chunk_gorny)
                if chunk_dolny is not None:
                    chunks.append(chunk_dolny)
                
                # Jeśli obrazek jest na początku, tylko chunk dolny
                # Jeśli obrazek jest na końcu, tylko chunk górny
                if not chunks:  # Brak sąsiadujących chunków
                    chunks = [None, None]
                elif len(chunks) == 1:
                    if chunk_gorny is None:  # Obrazek na początku
                        chunks = [None, chunks[0]]
                    else:  # Obrazek na końcu
                        chunks = [chunks[0], None]
                
                result.append({image_path: chunks})
        
        return result
    
    def process_file_to_json(self, file_path):
        """
        Przetwarza plik i zwraca JSON z obrazkami i kontekstem.
        
        Args:
            file_path (str): Ścieżka do pliku txt
            
        Returns:
            list: JSON z obrazkami i sąsiednimi chunkami
        """
        texts = self.process_file(file_path)
        return self.get_images_with_context_json(texts)
    
    def process_text_to_json(self, content):
        """
        Przetwarza tekst i zwraca JSON z obrazkami i kontekstem.
        
        Args:
            content (str): Tekst do przetworzenia
            
        Returns:
            list: JSON z obrazkami i sąsiednimi chunkami
        """
        texts = self.process_text(content)
        return self.get_images_with_context_json(texts)
    
    def save_to_json(self, texts, output_file="images_context.json"):
        """
        Zapisuje przetworzoną listę do pliku JSON.
        
        Args:
            texts (list): Przetworzona lista z wynikami process_file/process_text
            output_file (str): Ścieżka gdzie zapisać plik JSON
            
        Returns:
            str: Ścieżka do utworzonego pliku lub None jeśli wystąpił błąd
        """
        try:
            result = self.get_images_with_context_json(texts)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Wyniki zapisane do pliku: {output_file}")
            print(f"Liczba obrazków: {len(result)}")
            return output_file
            
        except Exception as e:
            print(f"Błąd podczas zapisywania do JSON: {e}")
            return None
    
    def save_file_to_json(self, file_path, output_file="images_context.json"):
        """
        Przetwarza plik i zapisuje wyniki do JSON.
        
        Args:
            file_path (str): Ścieżka do pliku txt
            output_file (str): Nazwa pliku wyjściowego JSON
            
        Returns:
            str: Ścieżka do utworzonego pliku lub None jeśli wystąpił błąd
        """
        try:
            texts = self.process_file(file_path)
            return self.save_to_json(texts, output_file)
            
        except Exception as e:
            print(f"Błąd podczas przetwarzania pliku: {e}")
            return None
        

class ImageContextFilter:
    def __init__(self, embedder=None):
        """
        Inicjalizuje filtr kontekstu obrazów.
        
        Args:
            embedder: Instancja CLIPEmbedder (jeśli None, zostanie utworzona nowa)
        """
        if embedder is None:
            self.embedder = CLIPEmbedder()
        else:
            self.embedder = embedder
    
    def load_images_context(self, json_path):
        """
        Wczytuje dane z pliku JSON z kontekstem obrazów.
        
        Args:
            json_path (str): Ścieżka do pliku JSON
            
        Returns:
            list: Wczytane dane JSON
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku JSON: {e}")
            return []
    
    def calculate_cosine_similarity(self, embedding1, embedding2):
        """
        Oblicza podobieństwo cosinusowe między dwoma embeddingami.
        
        Args:
            embedding1 (np.array): Pierwszy embedding
            embedding2 (np.array): Drugi embedding
            
        Returns:
            float: Wartość podobieństwa cosinusowego
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            # Sprawdź czy embeddingi mają te same wymiary
            if embedding1.shape != embedding2.shape:
                print(f"Różne wymiary embeddingów: {embedding1.shape} vs {embedding2.shape}")
                return 0.0
            
            # Reshape embeddingów do 2D dla sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            return cosine_similarity(emb1, emb2)[0][0]
        except Exception as e:
            print(f"Błąd podczas obliczania podobieństwa: {e}")
            return 0.0
    
    def filter_image_context(self, image_path, context_chunks):
        """
        Filtruje kontekst obrazu, usuwając chunk o niższym podobieństwie.
        
        Args:
            image_path (str): Ścieżka do obrazu
            context_chunks (list): Lista chunków kontekstu [chunk_górny, chunk_dolny]
            
        Returns:
            list: Przefiltrowana lista chunków (jeden chunk lub pusta lista)
        """
        # Sprawdź czy są jakieś chunki do porównania
        valid_chunks = [chunk for chunk in context_chunks if chunk is not None]
        
        if len(valid_chunks) == 0:
            return []
        elif len(valid_chunks) == 1:
            return valid_chunks
        
        try:
            # Sprawdź czy plik obrazu istnieje
            if not os.path.exists(image_path):
                print(f"Plik obrazu nie istnieje: {image_path}")
                return valid_chunks  # Zwróć wszystkie chunki jeśli obrazu nie ma
            
            # Wygeneruj embedding obrazu
            image_embedding = self.embedder.get_image_embedding(image_path)
            
            if image_embedding is None:
                print(f"Nie udało się wygenerować embeddingu dla obrazu: {image_path}")
                return valid_chunks
            
            # Oblicz podobieństwa dla każdego chunka
            similarities = []
            for chunk in context_chunks:
                if chunk is not None:
                    chunk_embedding = self.embedder.get_text_embedding(chunk)
                    if chunk_embedding is not None:
                        similarity = self.calculate_cosine_similarity(image_embedding, chunk_embedding)
                        similarities.append((chunk, similarity))
                        print(f"Podobieństwo dla chunka: {similarity:.4f}")
                    else:
                        similarities.append((chunk, 0.0))
                else:
                    similarities.append((None, -1.0))  # Niska wartość dla None
            
            # Znajdź chunk z najwyższym podobieństwem
            valid_similarities = [(chunk, sim) for chunk, sim in similarities if chunk is not None]
            
            if valid_similarities:
                best_chunk = max(valid_similarities, key=lambda x: x[1])
                print(f"Najlepszy chunk ma podobieństwo: {best_chunk[1]:.4f}")
                return [best_chunk[0]]
            else:
                return []
                
        except Exception as e:
            print(f"Błąd podczas przetwarzania obrazu {image_path}: {e}")
            return valid_chunks  # Zwróć wszystkie chunki w przypadku błędu
    
    def process_images_context(self, json_data):
        """
        Przetwarza wszystkie obrazy z JSON i filtruje ich kontekst.
        
        Args:
            json_data (list): Dane z pliku JSON
            
        Returns:
            list: Przefiltrowane dane z najlepszymi chunkami dla każdego obrazu
        """
        filtered_data = []
        
        for item in json_data:
            for image_path, context_chunks in item.items():
                print(f"\nPrzetwarzanie obrazu: {image_path}")
                
                # Filtruj kontekst
                best_chunk = self.filter_image_context(image_path, context_chunks)
                
                # Dodaj wynik do przefiltrowanych danych
                filtered_data.append({image_path: best_chunk})
                
                print(f"Wynik dla {image_path}: {len(best_chunk)} chunków")
        
        return filtered_data
    
    def save_filtered_context(self, filtered_data, output_path="filtered_images_context.json"):
        """
        Zapisuje przefiltrowane dane do pliku JSON.
        
        Args:
            filtered_data (list): Przefiltrowane dane
            output_path (str): Ścieżka do pliku wyjściowego
            
        Returns:
            str: Ścieżka do utworzonego pliku lub None jeśli wystąpił błąd
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            print(f"Przefiltrowane wyniki zapisane do pliku: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Błąd podczas zapisywania przefiltrowanych danych: {e}")
            return None
    
    def process_file(self, json_path, output_path="filtered_images_context.json"):
        """
        Kompletny proces filtrowania kontekstu obrazów z pliku JSON.
        
        Args:
            json_path (str): Ścieżka do pliku JSON z kontekstem
            output_path (str): Ścieżka do pliku wyjściowego
            
        Returns:
            str: Ścieżka do utworzonego pliku lub None jeśli wystąpił błąd
        """
        try:
            # Wczytaj dane
            json_data = self.load_images_context(json_path)
            if not json_data:
                return None
            
            # Przefiltruj kontekst
            filtered_data = self.process_images_context(json_data)
            
            # Zapisz wyniki
            return self.save_filtered_context(filtered_data, output_path)
            
        except Exception as e:
            print(f"Błąd podczas przetwarzania pliku: {e}")
            return None