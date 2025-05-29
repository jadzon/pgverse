import re
import os
import json
from pathlib import Path
from .chunker import TextChunker
from .embeddings import CLIPEmbedder
from sklearn.metrics.pairwise import cosine_similarity

class ImageTextProcessor:
    def __init__(self, chunker=None, max_tokens=150):
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
        self.file_directory = None  # Przechowuje folder z aktualnie przetwarzanym plikiem
    
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
        
        # Zapisz folder z plikiem txt do późniejszego użycia
        self.file_directory = Path(file_path).parent
        
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
                # Znajdź pozycję obrazka
                image_start = current_position + match.start()
                
                # Tekst przed obrazkiem
                text_before = content[current_position:image_start].strip()
                
                # Pochunkuj tekst jeśli istnieje
                if text_before:
                    chunks = self.chunker.chunk_text(text_before, max_tokens=self.max_tokens)
                    texts.extend(chunks)
                
                # Dodaj obrazek
                image_tag = f"<image/{match.group(1)}>"
                texts.append(image_tag)
                
                # Przesuń pozycję za obrazek
                current_position = image_start + len(match.group(0))
            else:
                # Brak więcej obrazków - pochunkuj resztę tekstu
                remaining_text = content[current_position:].strip()
                if remaining_text:
                    chunks = self.chunker.chunk_text(remaining_text, max_tokens=self.max_tokens)
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
        Wyciąga ścieżkę z elementu obrazka i konwertuje ją na ścieżkę relatywną względem pgverse.
        
        Args:
            image_element (str): Element obrazka w formacie <image/path>
            
        Returns:
            str: Ścieżka relatywna względem pgverse lub None jeśli element nie jest obrazkiem
        """
        if self.get_element_type(image_element) == 'image':
            match = re.match(r'<image/([^>]+)>', image_element)
            if match:
                relative_path = match.group(1)
                # Konwertuj na absolutną ścieżkę względem folderu z plikiem txt
                if self.file_directory:
                    absolute_path = self.file_directory / relative_path
                    
                    # Znajdź folder pgverse w ścieżce i utnij wszystko przed nim
                    absolute_str = str(absolute_path)
                    pgverse_index = absolute_str.find('pgverse')
                    
                    if pgverse_index != -1:
                        # Zwróć ścieżkę zaczynającą się od pgverse
                        return absolute_str[pgverse_index:]
                    else:
                        # Jeśli nie znaleziono pgverse, zwróć pełną ścieżkę
                        return absolute_str
                else:
                    return relative_path
        return None
    
    def get_images_with_context_json(self, texts):
        """
        Zwraca JSON z obrazkami i ich sąsiednimi chunkami.
        Dla obrazów bez bezpośredniego kontekstu znajduje najbliższy chunk.
        
        Args:
            texts (list): Tablica z wynikami process_file/process_text
            
        Returns:
            list: Lista słowników w formacie [{"path": ["chunk górny", "chunk dolny"]}]
        """
        result = []
        
        # Najpierw znajdź wszystkie chunki tekstowe
        text_chunks = []
        text_positions = []
        for i, element in enumerate(texts):
            if self.get_element_type(element) == 'text':
                text_chunks.append(element)
                text_positions.append(i)
        
        for i, element in enumerate(texts):
            if self.get_element_type(element) == 'image':
                image_path = self.get_image_path(element)
                
                # Znajdź sąsiednie chunki
                chunk_before = None
                chunk_after = None
                
                # Chunk przed obrazem
                if i > 0 and self.get_element_type(texts[i-1]) == 'text':
                    chunk_before = texts[i-1]
                
                # Chunk po obrazie
                if i < len(texts) - 1 and self.get_element_type(texts[i+1]) == 'text':
                    chunk_after = texts[i+1]
                
                # Jeśli brak bezpośredniego kontekstu, znajdź najbliższy
                if chunk_before is None and chunk_after is None:
                    closest_chunk = self._find_closest_text_chunk(i, text_positions, texts, text_chunks)
                    if closest_chunk:
                        chunk_before = closest_chunk
                
                # Utwórz entry dla obrazu
                context_list = []
                if chunk_before:
                    context_list.append(chunk_before)
                if chunk_after:
                    context_list.append(chunk_after)
                
                if image_path:
                    result.append({image_path: context_list})
        
        return result
    
    def _find_closest_text_chunk(self, image_position, text_positions, texts, text_chunks):
        """
        Znajduje najbliższy chunk tekstowy dla obrazu bez bezpośredniego kontekstu.
        
        Args:
            image_position (int): Pozycja obrazu w tablicy texts
            text_positions (list): Lista pozycji chunków tekstowych
            texts (list): Pełna tablica elementów
            text_chunks (list): Lista chunków tekstowych
            
        Returns:
            str: Najbliższy chunk tekstowy lub None
        """
        if not text_positions:
            return None
        
        # Znajdź pozycję tekstu najbliższą do obrazu
        distances = []
        for pos in text_positions:
            distance = abs(pos - image_position)
            distances.append((distance, pos))
        
        # Sortuj po odległości i wybierz najbliższy
        distances.sort(key=lambda x: x[0])
        closest_position = distances[0][1]
        
        # Znajdź chunk na tej pozycji
        for i, element in enumerate(texts):
            if i == closest_position and self.get_element_type(element) == 'text':
                return element
        
        return None
    
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
            json_data = self.get_images_with_context_json(texts)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            return output_file
        except Exception as e:
            print(f"Błąd zapisywania do JSON: {e}")
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
            json_data = self.process_file_to_json(file_path)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            return output_file
        except Exception as e:
            print(f"Błąd przetwarzania pliku: {e}")
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
            print(f"Błąd wczytywania JSON: {e}")
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
            return cosine_similarity([embedding1], [embedding2])[0][0]
        except Exception as e:
            print(f"Błąd obliczania podobieństwa: {e}")
            return 0.0
    
    def filter_image_context(self, image_path, context_chunks):
        """
        Filtruje kontekst obrazu, usuwając chunk o niższym podobieństwie.
        Jeśli obraz ma tylko jeden chunk (przypadek obrazów bez bezpośredniego kontekstu),
        zachowuje go.
        
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
            # Pobierz embedding obrazu
            image_embedding = self.embedder.get_image_embedding(image_path)
            if image_embedding is None:
                print(f"Błąd podczas generowania embeddingu obrazu {image_path}")
                return valid_chunks[:1]  # Zwróć pierwszy chunk jeśli nie ma embeddingu
            
            # Oblicz podobieństwa dla każdego chunku
            similarities = []
            for chunk in valid_chunks:
                text_embedding = self.embedder.get_text_embedding(chunk)
                similarity = self.calculate_cosine_similarity(image_embedding, text_embedding)
                similarities.append((similarity, chunk))
            
            # Posortuj po podobieństwie i zwróć najlepszy
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [similarities[0][1]]
                
        except Exception as e:
            print(f"Błąd filtrowania kontekstu dla {image_path}: {e}")
            return valid_chunks[:1]  # Zwróć pierwszy chunk w przypadku błędu
    
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
                filtered_chunks = self.filter_image_context(image_path, context_chunks)
                if filtered_chunks:
                    filtered_data.append({image_path: filtered_chunks})
        
        return filtered_data
    
    def save_filtered_context(self, filtered_data, output_path="filtered_images_context.json"):
        """
        Zapisuje przefiltrowane dane do pliku JSON.
        
        Args:
            filtered_data (list): Przefiltrowane dane
            output_path (str): Ścieżka do pliku wyjściowego
            
        Returns:
            bool: True jeśli zapisano pomyślnie, False w przeciwnym przypadku
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Błąd zapisywania przefiltrowanych danych: {e}")
            return False
    
    def process_file(self, json_path, output_path="filtered_images_context.json"):
        """
        Kompletny proces filtrowania kontekstu obrazów z pliku JSON.
        
        Args:
            json_path (str): Ścieżka do pliku JSON z kontekstem
            output_path (str): Ścieżka do pliku wyjściowego
            
        Returns:
            bool: True jeśli przetworzono pomyślnie, False w przeciwnym przypadku
        """
        try:
            json_data = self.load_images_context(json_path)
            filtered_data = self.process_images_context(json_data)
            return self.save_filtered_context(filtered_data, output_path)
        except Exception as e:
            print(f"Błąd przetwarzania pliku: {e}")
            return False

def main():
    """
    Główna funkcja demonstrująca użycie ImageTextProcessor
    """
    try:
        # Inicjalizuj procesor
        processor = ImageTextProcessor()
        
        # Sprawdź czy plik sample.txt istnieje
        if not os.path.exists("sample.txt"):
            print("Plik sample.txt nie istnieje. Tworzę przykładowy plik...")
            with open("sample.txt", "w", encoding="utf-8") as f:
                f.write("To jest przykładowy tekst przed obrazkiem.\n<image/figury/obraz1.jpg>\nTo jest tekst po pierwszym obrazie.\n<image/wzory/obraz2.png>\nTo jest tekst na końcu.")
        
        # Przetwórz plik sample.txt bez zapisywania pośredniego JSON
        print("Przetwarzanie pliku sample.txt...")
        texts = processor.process_file("sample.txt")
        json_data = processor.get_images_with_context_json(texts)
        
        if json_data:
            print(f"Znaleziono {len(json_data)} obrazów z kontekstem")
            for item in json_data:
                print(f"Obraz: {list(item.keys())[0]}")
                print(f"Kontekst: {list(item.values())[0]}")
        else:
            print("Nie znaleziono obrazów w pliku")
            
    except Exception as e:
        print(f"Błąd w funkcji main: {e}")

if __name__ == "__main__":
    main()