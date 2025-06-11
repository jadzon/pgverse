import re
import os
import json
from pathlib import Path
from .chunker import TextChunker
from .embeddings import CLIPEmbedder
from .latex import FormulaEmbedder
from sklearn.metrics.pairwise import cosine_similarity
import base64
from PIL import Image
import io
import numpy as np

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
        
        # ZMIANA: Użyj singletona zamiast tworzenia nowej instancji
        self.embedder = None  # Będzie inicjalizowany lazy loading
    
    def _get_embedder(self):
        """Lazy loading embeddera - tworzy tylko gdy potrzebny"""
        if self.embedder is None:
            self.embedder = CLIPEmbedder.get_instance()
        return self.embedder

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
        Określa typ elementu w tablicy - ROZSZERZONA WERSJA.
        POPRAWKA: Bardziej elastyczne wykrywanie typów.
        
        Args:
            element (str): Element z tablicy texts
            
        Returns:
            str: 'image', 'table', 'formula' lub 'text'
        """
        if not isinstance(element, str):
            return 'text'  # Domyślnie tekst dla nieprawidłowych typów
        
        element = element.strip()
        
        # Najpierw sprawdź czy to tag obrazu
        if not (element.startswith('<image/') and element.endswith('>')):
            return 'text'
        
        # POPRAWKA: Sprawdź różne możliwe ścieżki
        element_lower = element.lower()
        
        # Wykryj wzory (najwyższy priorytet)
        if any(keyword in element_lower for keyword in ['wzory', 'formula', 'formulas']):
            return 'formula'
        
        # Wykryj tabele  
        elif any(keyword in element_lower for keyword in ['tabele', 'table', 'tables']):
            return 'table'
        
        # Wykryj figury/obrazy
        elif any(keyword in element_lower for keyword in ['figury', 'figure', 'figures', 'image', 'images', 'obrazy']):
            return 'image'
        
        # Jeśli nie można określić typu, domyślnie image
        else:
            return 'image'
    
    def _get_element_path(self, element, element_type):
        """
        Uniwersalna metoda do pobierania ścieżki z różnych typów elementów.
        
        Args:
            element (str): Element z tablicy texts
            element_type (str): Typ elementu ('image', 'formula', 'table')
            
        Returns:
            str: Ścieżka do elementu lub None
        """
        # POPRAWKA: Wszystkie typy używają tego samego formatu <image/path>
        return self.get_image_path(element)
    
    def get_image_path(self, image_element):
        """
        Wyciąga ścieżkę z elementu obrazka i konwertuje ją na ścieżkę relatywną względem pgverse.
        POPRAWKA: Obsługuj wszystkie typy obrazów (figury, wzory, tabele).
        
        Args:
            image_element (str): Element obrazka w formacie <image/path>
            
        Returns:
            str: Ścieżka relatywna względem pgverse lub None jeśli element nie jest obrazkiem
        """
        element_type = self.get_element_type(image_element)
        
        # POPRAWKA: Akceptuj wszystkie typy obrazów
        if element_type in ['image', 'formula', 'table']:
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
    
    def get_images_with_context_json(self, texts, use_vision):
            
        if use_vision:
            return self._get_images_with_context_vision_json(texts)
        else:
            return self._get_images_with_context_embedding_json(texts)
    
    def _get_images_with_context_vision_json(self, texts):
        """
        NOWA implementacja dla vision system.
        Ekstraktuje ścieżki obrazów z texts jako listę stringów.
        
        Args:
            texts (list): Tablica z wynikami process_file/process_text
            
        Returns:
            list: Lista stringów ze ścieżkami do obrazów
        """
        print("Ekstraktuję ścieżki obrazów dla vision system...")
        
        image_paths = []
        
        for i, element in enumerate(texts):
            element_type = self.get_element_type(element)
            
            # Sprawdź czy to element obrazu (image, formula, table)
            if element_type in ['image', 'formula', 'table']:
                print(f"Znaleziono element typu '{element_type}': {element}")
                
                # Użyj uniwersalnej metody do pobierania ścieżki
                image_path = self._get_element_path(element, element_type)
                
                if image_path:
                    # Sprawdź czy obraz istnieje przed dodaniem do listy
                    absolute_path = self._resolve_image_path(image_path)
                    if os.path.exists(absolute_path):
                        image_paths.append(image_path)
                        print(f"Dodano ścieżkę: {image_path}")
                    else:
                        print(f"Pomijam nieistniejący obraz: {image_path} (szukano w: {absolute_path})")
                else:
                    print(f"Nie udało się wyciągnąć ścieżki z elementu: {element}")
        
        print(f"Wyekstraktowano {len(image_paths)} ścieżek obrazów:")
        for path in image_paths:
            print(f"  - {path}")
        
        # Eksportuj listę ścieżek do JSON
        exported_data = []
        # exported_data = self._export_vision_paths_to_json()
        return exported_data

    def _export_vision_paths_to_json(self):
        """
        Eksportuje listę ścieżek obrazów do pliku folderOCR_context.json.
        
        Args:
            image_paths (list): Lista ścieżek do obrazów
        """
        # exported_data = VisionContext.get_json_data()

        # return exported_data
    

    def _get_images_with_context_embedding_json(self, texts):
        """
        Zwraca JSON z obrazkami i ich najlepszym chunkiem na podstawie podobieństwa embeddingowego.
        NOWE: Używa embeddingów do wyboru najlepszego chunku dla każdego obrazu.
        ULEPSZONE: Obsługuje wzory matematyczne z embeddingami LaTeX.
        
        Args:
            texts (list): Tablica z wynikami process_file/process_text
            
        Returns:
            list: Lista słowników w formacie [{"path": ["najlepszy chunk"]}]
        """


        result = []
        
        # Najpierw znajdź wszystkie chunki tekstowe
        text_chunks = []
        text_positions = []
        for i, element in enumerate(texts):
            if self.get_element_type(element) == 'text':
                text_chunks.append(element)
                text_positions.append(i)
        
        print(f"Znaleziono {len(text_chunks)} chunków tekstowych")
        
        for i, element in enumerate(texts):
            element_type = self.get_element_type(element)
            
            # POPRAWKA: Obsługuj wszystkie typy obrazów (image, formula, table)
            if element_type in ['image', 'formula', 'table']:
                print(f"Przetwarzam element typu '{element_type}': {element}")
                
                # POPRAWKA: Użyj uniwersalnej metody do pobierania ścieżki
                image_path = self._get_element_path(element, element_type)
                
                if not image_path:
                    print(f"Nie udało się wyciągnąć ścieżki z elementu: {element}")
                    continue
                
                # Sprawdź czy obraz istnieje przed dodaniem do JSON
                absolute_path = self._resolve_image_path(image_path)
                if not os.path.exists(absolute_path):
                    print(f"Pomijam nieistniejący obraz: {image_path} (szukano w: {absolute_path})")
                    continue  # Pomiń ten obraz jeśli nie istnieje
                
                print(f"Obraz istnieje: {absolute_path}")
                
                # Zbierz potencjalne chunki kontekstu
                potential_chunks = []
                
                # Chunk przed obrazem
                if i > 0 and self.get_element_type(texts[i-1]) == 'text':
                    potential_chunks.append(texts[i-1])
                    print(f"Dodano chunk przed obrazem: {texts[i-1][:50]}...")
                
                # Chunk po obrazie
                if i < len(texts) - 1 and self.get_element_type(texts[i+1]) == 'text':
                    potential_chunks.append(texts[i+1])
                    print(f"Dodano chunk po obrazie: {texts[i+1][:50]}...")
                
                # Jeśli brak bezpośredniego kontekstu, znajdź najbliższy
                if not potential_chunks:
                    closest_chunk = self._find_closest_text_chunk(i, text_positions, texts, text_chunks)
                    if closest_chunk:
                        potential_chunks.append(closest_chunk)
                        print(f"Dodano najbliższy chunk: {closest_chunk[:50]}...")
                
                print(f"Znaleziono {len(potential_chunks)} potencjalnych chunków kontekstu")
                
                # Wybierz najlepszy chunk na podstawie podobieństwa embeddingowego
                if potential_chunks and image_path:
                    if element_type == 'formula':
                        # NOWE: Użyj embeddingów wzorów dla formuł
                        print(f"Używam embeddingów LaTeX dla wzoru: {image_path}")
                        best_chunk = self._select_best_chunk_for_formula(absolute_path, potential_chunks)
                    else:
                        # Standardowe obrazy i tabele - użyj CLIP
                        print(f"Używam embeddingów CLIP dla obrazu/tabeli: {image_path}")
                        best_chunk = self._select_best_chunk_by_embedding(absolute_path, potential_chunks)
                    
                    if best_chunk:
                        result.append({image_path: [best_chunk]})
                        print(f"Dodano do rezultatu: {image_path} -> {best_chunk[:50]}...")
                    else:
                        print(f"Nie udało się wybrać najlepszego chunku dla: {image_path}")
                else:
                    print(f"Brak chunków kontekstu dla: {image_path}")
    
        print(f"Zwracam {len(result)} elementów w JSON")
        return result
    
    def _select_best_chunk_for_formula(self, image_path, chunks):
        """
        Wybiera najlepszy chunk dla wzoru matematycznego używając embeddingów LaTeX.
        
        Args:
            image_path (str): Absolutna ścieżka do obrazu wzoru
            chunks (list): Lista chunków do porównania
            
        Returns:
            str: Najlepszy chunk lub None jeśli wystąpił błąd
        """
        try:
            print(f"Próbuję znaleźć embedding LaTeX dla wzoru: {image_path}")
            
            # Znajdź odpowiadający plik PNG i JSON dla wzoru
            png_filename = Path(image_path).name
            print(f"Nazwa pliku PNG: {png_filename}")
            
            # POPRAWKA: Lepsze szukanie pliku latex_wzory.json
            # Sprawdź różne możliwe lokalizacje
            possible_json_paths = []
            
            # 1. W folderze wzory obok detekcje
            if self.file_directory:
                parent_folder = self.file_directory.parent  # Wyjdź z detekcje
                wzory_dir = parent_folder / "wzory"
                possible_json_paths.append(wzory_dir / "latex_wzory.json")
            
            # 2. W folderze rezultaty/wzory
            if self.file_directory:
                parent_folder = self.file_directory.parent
                rezultaty_wzory = parent_folder / "rezultaty" / "wzory"
                possible_json_paths.append(rezultaty_wzory / "latex_wzory.json")
            
            # 3. W tym samym folderze co obraz
            image_dir = Path(image_path).parent
            possible_json_paths.append(image_dir / "latex_wzory.json")
            
            # 4. Jeden poziom wyżej od obrazu
            possible_json_paths.append(image_dir.parent / "latex_wzory.json")
            
            json_path = None
            for path in possible_json_paths:
                print(f"Sprawdzam: {path}")
                if path.exists():
                    json_path = path
                    print(f"Znaleziono latex_wzory.json w: {json_path}")
                    break
            
            if not json_path:
                print(f"Nie znaleziono pliku latex_wzory.json w żadnej z lokalizacji")
                return self._fallback_to_clip_for_formula(image_path, chunks)
            
            # Pobierz embedding wzoru przez statyczną metodę
            formula_embedding = FormulaEmbedder.get_formula_embedding_from_paths(
                str(json_path), png_filename
            )
            
            if formula_embedding is None:
                print(f"Nie udało się uzyskać embeddingu wzoru dla {png_filename}")
                return self._fallback_to_clip_for_formula(image_path, chunks)
            
            print(f"Uzyskano embedding LaTeX o wymiarach: {formula_embedding.shape}")
            
            # Użyj CLIP do embeddingu tekstu (zachowujemy kompatybilność wymiarów)
            embedder = self._get_embedder()
            
            # Oblicz podobieństwa dla każdego chunku
            similarities = []
            for chunk in chunks:
                if chunk and chunk.strip():
                    text_embedding = embedder.get_text_embedding(chunk)
                    if text_embedding is not None:
                        # Normalizuj embeddingi do porównania
                        formula_emb_norm = formula_embedding / (np.linalg.norm(formula_embedding) + 1e-8)
                        text_emb_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
                        
                        # Oblicz podobieństwo - użyj pierwszych 512 wymiarów wzoru
                        if len(formula_emb_norm) >= 512:
                            formula_emb_truncated = formula_emb_norm[:512]
                        else:
                            # Jeśli embedding wzoru ma mniej niż 512 wymiarów, dopełnij zerami
                            formula_emb_truncated = np.pad(formula_emb_norm, 
                                                         (0, max(0, 512 - len(formula_emb_norm))), 
                                                         'constant')
                        
                        similarity = self._calculate_cosine_similarity(formula_emb_truncated, text_emb_norm)
                        similarities.append((similarity, chunk))
                        print(f"Podobieństwo wzoru '{png_filename}' do chunku '{chunk[:50]}...': {similarity:.4f}")
            
            if not similarities:
                print(f"Brak prawidłowych embeddingów dla chunków wzoru {png_filename}")
                return chunks[0] if chunks else None
            
            # Posortuj po podobieństwie i zwróć najlepszy
            similarities.sort(key=lambda x: x[0], reverse=True)
            best_similarity, best_chunk = similarities[0]
            print(f"Wybrano najlepszy chunk dla wzoru (podobieństwo: {best_similarity:.4f}): {best_chunk[:50]}...")
            
            return best_chunk
                
        except Exception as e:
            print(f"Błąd podczas wyboru najlepszego chunku dla wzoru {image_path}: {e}")
            return self._fallback_to_clip_for_formula(image_path, chunks)
    
    def _fallback_to_clip_for_formula(self, image_path, chunks):
        """
        Fallback do CLIP gdy embedding wzoru nie jest dostępny.
        
        Args:
            image_path (str): Ścieżka do obrazu wzoru
            chunks (list): Lista chunków
            
        Returns:
            str: Najlepszy chunk według CLIP
        """
        print(f"Używam fallback CLIP dla wzoru: {image_path}")
        return self._select_best_chunk_by_embedding(image_path, chunks)
    
    def _select_best_chunk_by_embedding(self, image_path, chunks):
        """
        Wybiera najlepszy chunk na podstawie podobieństwa embeddingowego z obrazem.
        
        Args:
            image_path (str): Absolutna ścieżka do obrazu
            chunks (list): Lista chunków do porównania
            
        Returns:
            str: Najlepszy chunk lub None jeśli wystąpił błąd
        """
        try:
            # ZMIANA: Użyj singletona
            embedder = self._get_embedder()
            
            # Pobierz embedding obrazu
            image_embedding = embedder.get_image_embedding(image_path)
            if image_embedding is None:
                print(f"Błąd podczas generowania embeddingu obrazu {image_path}")
                return chunks[0] if chunks else None  # Zwróć pierwszy chunk jeśli nie ma embeddingu
            
            # Oblicz podobieństwa dla każdego chunku
            similarities = []
            for chunk in chunks:
                if chunk and chunk.strip():  # Sprawdź czy chunk nie jest pusty
                    text_embedding = embedder.get_text_embedding(chunk)
                    if text_embedding is not None:
                        similarity = self._calculate_cosine_similarity(image_embedding, text_embedding)
                        similarities.append((similarity, chunk))
                        print(f"Podobieństwo dla chunku '{chunk[:50]}...': {similarity:.4f}")
            
            if not similarities:
                print(f"Brak prawidłowych embeddingów dla chunków obrazu {image_path}")
                return chunks[0] if chunks else None
            
            # Posortuj po podobieństwie i zwróć najlepszy
            similarities.sort(key=lambda x: x[0], reverse=True)
            best_similarity, best_chunk = similarities[0]
            print(f"Wybrano najlepszy chunk (podobieństwo: {best_similarity:.4f}): {best_chunk[:50]}...")
            
            return best_chunk
                
        except Exception as e:
            print(f"Błąd podczas wyboru najlepszego chunku dla {image_path}: {e}")
            return chunks[0] if chunks else None  # Zwróć pierwszy chunk w przypadku błędu
    
    def _calculate_cosine_similarity(self, embedding1, embedding2):
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

    def _resolve_image_path(self, image_path):
        """
        Konwertuje ścieżkę obrazu na absolutną.
        ULEPSZONE: Lepsze wyszukiwanie obrazów w różnych folderach.
        
        Args:
            image_path (str): Ścieżka obrazu (może być relatywna lub absolutna)
            
        Returns:
            str: Absolutna ścieżka do obrazu
        """
        path_obj = Path(image_path)
        
        # Jeśli już jest absolutna i istnieje
        if path_obj.is_absolute() and path_obj.exists():
            return str(path_obj)
        
        # SPRAWDŹ: Ścieżkę relatywną względem folderu z plikiem txt
        if self.file_directory:
            # Sprawdź bezpośrednio w folderze z plikiem
            full_path = self.file_directory / image_path
            if full_path.exists():
                return str(full_path)
            
            # Sprawdź z normalizacją separatorów
            normalized_path = image_path.replace('\\', os.sep).replace('/', os.sep)
            full_path = self.file_directory / normalized_path
            if full_path.exists():
                return str(full_path)
            
            # NOWE: Sprawdź w folderach rezultaty/figury, rezultaty/wzory, rezultaty/tabele
            parent_folder = self.file_directory.parent  # Wyjdź z folderu detekcje
            rezultaty_path = parent_folder / "rezultaty"
            
            if rezultaty_path.exists():
                filename = Path(image_path).name
                # Sprawdź w różnych podfolderach rezultaty
                for subfolder in ["figury", "wzory", "tabele", "obrazy", "figures", "formulas", "tables"]:
                    potential_path = rezultaty_path / subfolder / filename
                    if potential_path.exists():
                        return str(potential_path)
                    
                    # Sprawdź rekursywnie w podfolderach
                    subfolder_path = rezultaty_path / subfolder
                    if subfolder_path.exists():
                        for found_file in subfolder_path.rglob(filename):
                            if found_file.exists():
                                return str(found_file)
        
        # Jeśli zaczyna się od 'pgverse', znajdź bazowy folder
        if image_path.startswith('pgverse'):
            # Znajdź folder pgverse w strukturze
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:  # Dopóki nie dojdziemy do roota
                if current_dir.name == 'pgverse':
                    # Usuń 'pgverse/' z początku ścieżki
                    relative_part = image_path[len('pgverse'):].lstrip('\\/')
                    full_path = current_dir / relative_part
                    if full_path.exists():
                        return str(full_path)
                    break
                current_dir = current_dir.parent
        
        # Ostatnia próba - zwróć oryginalną ścieżkę (może nie istnieć)
        return image_path

    def _find_closest_text_chunk(self, image_position, text_positions, texts, text_chunks):
        """
        Znajduje najbliższy chunk tekstowy do pozycji obrazu.
        
        Args:
            image_position (int): Pozycja obrazu w tablicy texts
            text_positions (list): Lista pozycji chunków tekstowych
            texts (list): Pełna tablica texts
            text_chunks (list): Lista samych chunków tekstowych
            
        Returns:
            str: Najbliższy chunk tekstowy lub None jeśli nie znaleziono
        """
        if not text_positions or not text_chunks:
            return None
        
        # Znajdź najbliższą pozycję tekstową
        min_distance = float('inf')
        closest_chunk = None
        
        for i, text_pos in enumerate(text_positions):
            distance = abs(image_position - text_pos)
            if distance < min_distance:
                min_distance = distance
                closest_chunk = text_chunks[i]
        
        return closest_chunk

    # POPRAWKA: Uspójnij metody get_*_path()
    def get_table_path(self, table_element):
        """
        Wyciąga ścieżkę z elementu tabeli.
        POPRAWKA: Używa tego samego formatu co get_image_path.
        
        Args:
            table_element (str): Element tabeli w formacie <image/path>
        
        Returns:
            str: Ścieżka do tabeli lub None jeśli element nie jest tabelą
        """
        # POPRAWKA: Używaj get_image_path dla spójności
        return self.get_image_path(table_element)

    def get_formula_path(self, formula_element):
        """
        Wyciąga ścieżkę z elementu wzoru.
        POPRAWKA: Używa tego samego formatu co get_image_path.
            
        Args:
            formula_element (str): Element wzoru w formacie <image/path>
            
        Returns:
            str: Ścieżka do wzoru lub None jeśli element nie jest wzorem
        """
        # POPRAWKA: Używaj get_image_path dla spójności
        return self.get_image_path(formula_element)

    # Pozostałe metody bez zmian...
    def image_to_base64(self, image_path):
        """
        Konwertuje obraz na base64.
        
        Args:
            image_path (str): Ścieżka do obrazu
            
        Returns:
            str: Obraz zakodowany w base64 lub None jeśli wystąpił błąd
        """
        try:
            # Sprawdź czy plik istnieje
            if not os.path.exists(image_path):
                print(f"Obraz nie istnieje: {image_path}")
                return None
            
            # Otwórz i konwertuj obraz
            with Image.open(image_path) as img:
                # Konwertuj na RGB jeśli obraz ma kanał alpha
                if img.mode in ('RGBA', 'LA'):
                    # Utwórz białe tło
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])  # Użyj kanału alpha jako maski
                    else:
                        background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Zapisz do bufora jako JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                
                # Konwertuj na base64
                base64_string = base64.b64encode(buffer.read()).decode('utf-8')
                return base64_string
                
        except Exception as e:
            print(f"Błąd konwersji obrazu {image_path} na base64: {e}")
            return None

    def create_output_txt_with_base64(self, texts, output_file="output_with_base64.txt"):
        """
        Tworzy wynikowy plik txt z obrazami zamienionymi na base64 i pochunkowanym tekstem.
        WAŻNE: Używa już pochunkowanych tekstów z tablicy texts (te same co w JSON).
        Format: <image/path/base64_data>
        NOWE: Pomija nieistniejące obrazy.
        POPRAWKA: Obsługuje wszystkie typy obrazów.
        
        Args:
            texts (list): Tablica z wynikami process_file/process_text (już pochunkowana)
            output_file (str): Ścieżka do pliku wyjściowego
            
        Returns:
            str: Ścieżka do utworzonego pliku lub None jeśli wystąpił błąd
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for element in texts:
                    element_type = self.get_element_type(element)
                    
                    # POPRAWKA: Obsługuj wszystkie typy obrazów
                    if element_type in ['image', 'formula', 'table']:
                        # Pobierz ścieżkę obrazu
                        image_path = self.get_image_path(element)
                        if image_path:
                            # Konwertuj relatywną ścieżkę na absolutną
                            absolute_path = self._resolve_image_path(image_path)
                            
                            # NOWE: Sprawdź czy obraz istnieje
                            if not os.path.exists(absolute_path):
                                print(f"Pomijam nieistniejący obraz w TXT: {image_path} (szukano w: {absolute_path})")
                                continue  # Pomiń nieistniejące obrazy
                            
                            # Konwertuj obraz na base64
                            base64_data = self.image_to_base64(absolute_path)
                            
                            if base64_data:
                                # Zapisz jako <image/path/base64_data> z pustą linią po nim
                                f.write(f"<image/{image_path}/{base64_data}>\n\n")
                            else:
                                # Jeśli konwersja się nie udała, zapisz oryginalną ścieżkę z komentarzem
                                f.write(f"<image/ERROR_CONVERTING:{image_path}>\n\n")
                    else:
                        # To jest chunk tekstowy - zapisz go z pustą linią po nim
                        # WAŻNE: To są już pochunkowane teksty (te same co w JSON)
                        f.write(f"{element}\n\n")
            
            return output_file
            
        except Exception as e:
            print(f"Błąd tworzenia pliku z base64: {e}")
            return None

    def process_file_to_txt_with_base64(self, file_path, output_file=None):
        """
        Przetwarza plik txt i tworzy nowy plik z obrazami jako base64.
        
        Args:
            file_path (str): Ścieżka do pliku wejściowego
            output_file (str): Ścieżka do pliku wyjściowego (domyślnie: input_file_base64.txt)
            
        Returns:
            str: Ścieżka do utworzonego pliku lub None jeśli wystąpił błąd
        """
        try:
            if output_file is None:
                input_path = Path(file_path)
                output_file = str(input_path.parent / f"{input_path.stem}_base64.txt")
            
            # Przetwórz plik
            texts = self.process_file(file_path)
            
            # Utwórz plik z base64
            return self.create_output_txt_with_base64(texts, output_file)
            
        except Exception as e:
            print(f"Błąd przetwarzania pliku do base64: {e}")
            return None
    
    def create_output_txt_chunks_only(self, texts, output_file="chunks_only.txt"):
        """
        Tworzy wynikowy plik txt z samymi chunkami tekstowymi (bez obrazów).
        Chunki są oddzielone pustą linią.
        
        Args:
            texts (list): Tablica z wynikami process_file/process_text
            output_file (str): Ścieżka do pliku wyjściowego
            
        Returns:
            str: Ścieżka do utworzonego pliku lub None jeśli wystąpił błąd
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for element in texts:
                    if self.get_element_type(element) == 'text':
                        # Zapisz tylko chunki tekstowe z pustą linią po nich
                        f.write(f"{element}\n\n")
            
            return output_file
            
        except Exception as e:
            print(f"Błąd tworzenia pliku z chunkami: {e}")
            return None

    def process_image_embedding(self, element):
        """
        Przetwarza element typu 'image' i generuje embedding wizualny.
        
        Args:
            element (str): Element obrazka w formacie <image/path>
            
        Returns:
            np.array: 512-wymiarowy embedding obrazu lub None
        """
        if self.get_element_type(element) == 'image':
            image_path = self.get_image_path(element)
            if image_path:
                # Rozwiąż ścieżkę do absolutnej
                absolute_path = self._resolve_image_path(image_path)
                
                # Sprawdź czy obraz istnieje
                if not os.path.exists(absolute_path):
                    print(f"Obraz nie istnieje: {absolute_path}")
                    return None
                
                # Użyj CLIP embeddera dla obrazów
                embedder = self._get_embedder()
                image_embedding = embedder.get_image_embedding(absolute_path)
                
                if image_embedding is not None:
                    print(f"Wygenerowano embedding obrazu: {image_path}")
                    return image_embedding
                else:
                    print(f"Błąd generowania embeddingu dla obrazu: {image_path}")
                    return None
        return None

    def process_text_embedding(self, element):
        """
        Przetwarza element typu 'text' i generuje embedding tekstowy.
        
        Args:
            element (str): Fragment tekstu
        
        Returns:
            np.array: 512-wymiarowy embedding tekstu lub None
        """
        if self.get_element_type(element) == 'text':
            # Sprawdź czy tekst nie jest pusty
            if not element or not element.strip():
                print("Pusty fragment tekstu")
                return None
            
            # Użyj CLIP embeddera dla tekstu
            embedder = self._get_embedder()
            text_embedding = embedder.get_text_embedding(element)
            
            if text_embedding is not None:
                print(f"Wygenerowano embedding tekstu: '{element[:50]}...'")
                return text_embedding
            else:
                print(f"Błąd generowania embeddingu dla tekstu")
                return None
        return None

    def process_table_embedding(self, element, external_embedding=None):
        """
        Przetwarza element typu 'table' używając zewnętrznego embeddingu.
        
        Args:
            element (str): Element tabeli w formacie <image/path>
            external_embedding (np.array): Gotowy embedding dostarczony z zewnątrz
        
        Returns:
            np.array: Specjalizowany embedding tabeli lub None
        """
        if self.get_element_type(element) == 'table':
            if external_embedding is not None:
                print(f"Otrzymano zewnętrzny embedding dla tabeli: {element}")
                print(f"Wymiary embeddingu: {external_embedding.shape}")
                return external_embedding
            else:
                print(f"Brak zewnętrznego embeddingu dla tabeli: {element}")
                return None
        return None
    
    def process_formula_embedding(self, element, external_embedding=None):
        """
        Przetwarza element typu 'formula' używając embeddingów LaTeX lub zewnętrznego embeddingu.
        ULEPSZONE: Integracja z FormulaEmbedder z latex.py
        
        Args:
            element (str): Element wzoru w formacie <image/path>
            external_embedding (np.array): Gotowy embedding dostarczony z zewnątrz (opcjonalny)
            
        Returns:
            np.array: Specjalizowany embedding wzoru lub None
        """
        if self.get_element_type(element) == 'formula':
            # Jeśli podano zewnętrzny embedding, użyj go
            if external_embedding is not None:
                print(f"Otrzymano zewnętrzny embedding dla wzoru: {element}")
                print(f"Wymiary embeddingu: {external_embedding.shape}")
                return external_embedding
            
            # Spróbuj wygenerować embedding przez LaTeX
            try:
                image_path = self.get_image_path(element)
                if image_path:
                    absolute_path = self._resolve_image_path(image_path)
                    png_filename = Path(absolute_path).name
                    
                    # POPRAWKA: Używaj tej samej logiki co w _select_best_chunk_for_formula
                    possible_json_paths = []
                    
                    # 1. W folderze wzory obok detekcje
                    if self.file_directory:
                        parent_folder = self.file_directory.parent
                        wzory_dir = parent_folder / "wzory"
                        possible_json_paths.append(wzory_dir / "latex_wzory.json")
                    
                    # 2. W folderze rezultaty/wzory
                    if self.file_directory:
                        parent_folder = self.file_directory.parent
                        rezultaty_wzory = parent_folder / "rezultaty" / "wzory"
                        possible_json_paths.append(rezultaty_wzory / "latex_wzory.json")
                    
                    # 3. W tym samym folderze co obraz
                    image_dir = Path(absolute_path).parent
                    possible_json_paths.append(image_dir / "latex_wzory.json")
                    
                    # 4. Jeden poziom wyżej od obrazu
                    possible_json_paths.append(image_dir.parent / "latex_wzory.json")
                    
                    json_path = None
                    for path in possible_json_paths:
                        if path.exists():
                            json_path = path
                            break
                    
                    if json_path and json_path.exists():
                        # Użyj statycznej metody z FormulaEmbedder
                        formula_embedding = FormulaEmbedder.get_formula_embedding_from_paths(
                            str(json_path), png_filename
                        )
                        
                        if formula_embedding is not None:
                            print(f"Wygenerowano embedding wzoru LaTeX: {element}")
                            print(f"Wymiary embeddingu: {formula_embedding.shape}")
                            return formula_embedding
                        else:
                            print(f"Nie udało się wygenerować embeddingu LaTeX dla: {png_filename}")
                    else:
                        print(f"Nie znaleziono pliku latex_wzory.json dla wzoru")
                
                # Fallback do CLIP jeśli LaTeX nie zadziałał
                print(f"Używam fallback CLIP dla wzoru: {element}")
                return self.process_image_embedding(element)
                        
            except Exception as e:
                print(f"Błąd podczas generowania embeddingu LaTeX dla wzoru: {e}")
                # Fallback do CLIP
                return self.process_image_embedding(element)
        
        return None


class ImageContextFilter:
    def __init__(self, embedder=None):
        """
        Inicjalizuje filtr kontekstu obrazów.
        
        Args:
            embedder: Instancja CLIPEmbedder (opcjonalna - użyje singletona)
        """
        # ZMIANA: Zawsze użyj singletona
        if embedder is None:
            self.embedder = CLIPEmbedder.get_instance()
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
        Filtruje kontekst obrazu używając odpowiednich embeddingów.
        ULEPSZONE: Obsługuje wzory matematyczne z embeddingami LaTeX.
        """
        valid_chunks = [chunk for chunk in context_chunks if chunk is not None]
        
        if len(valid_chunks) <= 1:
            return valid_chunks
        
        try:
            # Sprawdź czy to wzór matematyczny
            if 'wzory' in image_path.lower() or 'formula' in image_path.lower():
                return self._filter_formula_context(image_path, valid_chunks)
            else:
                return self._filter_image_context_clip(image_path, valid_chunks)
                
        except Exception as e:
            print(f"Błąd filtrowania kontekstu dla {image_path}: {e}")
            return valid_chunks[:1]  # Zwróć pierwszy chunk w przypadku błędu
    
    def _filter_formula_context(self, formula_path, chunks):
        """
        Filtruje kontekst wzoru używając embeddingów LaTeX.
        
        Args:
            formula_path (str): Ścieżka do obrazu wzoru
            chunks (list): Lista chunków kontekstu
            
        Returns:
            list: Najlepsze chunki dla wzoru
        """
        try:
            png_filename = Path(formula_path).name
            
            # POPRAWKA: Lepsze szukanie pliku latex_wzory.json - podobnie jak w ImageTextProcessor
            possible_json_paths = []
            
            # Różne możliwe lokalizacje
            image_dir = Path(formula_path).parent
            possible_json_paths.append(image_dir / "latex_wzory.json")
            possible_json_paths.append(image_dir.parent / "latex_wzory.json")
            possible_json_paths.append(image_dir.parent / "wzory" / "latex_wzory.json")
            possible_json_paths.append(image_dir.parent.parent / "wzory" / "latex_wzory.json")
            
            json_path = None
            for path in possible_json_paths:
                if path.exists():
                    json_path = path
                    break
            
            if not json_path:
                print(f"Nie znaleziono latex_wzory.json, używam CLIP fallback dla {formula_path}")
                return self._filter_image_context_clip(formula_path, chunks)
            
            # Pobierz embedding wzoru
            formula_embedding = FormulaEmbedder.get_formula_embedding_from_paths(
                str(json_path), png_filename
            )
            
            if formula_embedding is None:
                print(f"Nie udało się uzyskać embeddingu LaTeX, używam CLIP fallback dla {png_filename}")
                return self._filter_image_context_clip(formula_path, chunks)
            
            # Porównaj z chunkami tekstowymi przez CLIP
            similarities = []
            for chunk in chunks:
                text_embedding = self.embedder.get_text_embedding(chunk)
                if text_embedding is not None:
                    # Normalizuj embeddingi
                    formula_emb_norm = formula_embedding / (np.linalg.norm(formula_embedding) + 1e-8)
                    text_emb_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
                    
                    # Dopasuj wymiary (użyj pierwszych 512 wymiarów wzoru)
                    if len(formula_emb_norm) >= 512:
                        formula_emb_truncated = formula_emb_norm[:512]
                    else:
                        formula_emb_truncated = np.pad(formula_emb_norm, 
                                                     (0, max(0, 512 - len(formula_emb_norm))), 
                                                     'constant')
                    
                    similarity = self.calculate_cosine_similarity(formula_emb_truncated, text_emb_norm)
                    similarities.append((similarity, chunk))
            
            if not similarities:
                return chunks[:1]
            
            # Zwróć najlepszy chunk
            similarities.sort(key=lambda x: x[0], reverse=True)
            best_chunk = similarities[0][1]
            print(f"Wybrano najlepszy chunk dla wzoru {png_filename}: {best_chunk[:50]}...")
            
            return [best_chunk]
            
        except Exception as e:
            print(f"Błąd filtrowania kontekstu wzoru {formula_path}: {e}")
            return self._filter_image_context_clip(formula_path, chunks)
    
    def _filter_image_context_clip(self, image_path, chunks):
        """
        Filtruje kontekst obrazu używając embeddingów CLIP.
        
        Args:
            image_path (str): Ścieżka do obrazu
            chunks (list): Lista chunków kontekstu
            
        Returns:
            list: Najlepsze chunki dla obrazu
        """
        try:
            # Embedding obrazu przez CLIP
            image_embedding = self.embedder.get_image_embedding(image_path)
            if image_embedding is None:
                return chunks[:1]
            
            # Embeddingi tekstowe przez CLIP
            similarities = []
            for chunk in chunks:
                text_embedding = self.embedder.get_text_embedding(chunk)
                similarity = self.calculate_cosine_similarity(image_embedding, text_embedding)
                similarities.append((similarity, chunk))
            
            # Zwróć najlepszy chunk
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [similarities[0][1]]
            
        except Exception as e:
            print(f"Błąd filtrowania kontekstu CLIP dla {image_path}: {e}")
            return chunks[:1]
    
    def process_images_context(self, json_data):
        """
        Przetwarza wszystkie obrazy z JSON i filtruje ich kontekst.
        NOWE: Sprawdza czy obrazy istnieją przed przetwarzaniem.
        ULEPSZONE: Obsługuje wzory z embeddingami LaTeX.
        
        Args:
            json_data (list): Dane z pliku JSON
            
        Returns:
            list: Przefiltrowane dane z najlepszymi chunkami dla każdego obrazu
        """
        filtered_data = []
        
        for item in json_data:
            for image_path, context_chunks in item.items():
                # NOWE: Sprawdź czy obraz istnieje przed filtrowaniem
                if not os.path.exists(image_path):
                    # Jeśli ścieżka nie jest absolutna, spróbuj ją rozwiązać
                    absolute_path = self._resolve_image_path_for_filter(image_path)
                    if not os.path.exists(absolute_path):
                        print(f"Pomijam nieistniejący obraz w filtrze: {image_path}")
                        continue
                    else:
                        # Użyj absolutnej ścieżki do filtrowania
                        image_path = absolute_path
                
                filtered_chunks = self.filter_image_context(image_path, context_chunks)
                if filtered_chunks:
                    filtered_data.append({image_path: filtered_chunks})
        
        return filtered_data
    
    def _resolve_image_path_for_filter(self, image_path):
        """
        Rozwiązuje ścieżkę obrazu dla filtra kontekstu.
        Podobne do _resolve_image_path ale bardziej ogólne.
        
        Args:
            image_path (str): Ścieżka do obrazu
            
        Returns:
            str: Rozwiązana ścieżka
        """
        path_obj = Path(image_path)
        
        # Jeśli już jest absolutna
        if path_obj.is_absolute():
            return str(path_obj)
        
        # Znajdź folder pgverse w strukturze
        current_dir = Path(__file__).parent
        while current_dir.parent != current_dir:
            if current_dir.name == 'pgverse':
                if image_path.startswith('pgverse'):
                    relative_part = image_path[len('pgverse'):].lstrip('\\/')
                    full_path = current_dir / relative_part
                else:
                    full_path = current_dir / image_path
                return str(full_path)
            current_dir = current_dir.parent
        
        return image_path