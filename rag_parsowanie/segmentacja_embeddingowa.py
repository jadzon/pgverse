import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Dodajemy bibliotekę do modelu embeddingowego i zapisywania do pliku
from sentence_transformers import SentenceTransformer
import os

#########################
# Pomocnicze funkcje
#########################

def simple_recursive_split(text, target_tokens):
    """
    Dzieli tekst rekurencyjnie na fragmenty, które mają przybliżoną długość target_tokens.
    Używamy prostego podziału na zdania.
    """
    # Najpierw normalizujemy tekst
    text = re.sub(r'\s+', ' ', text.strip())
    # Dzielimy tekst na zdania (używamy prostego wyrażenia regularnego)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    for s in sentences:
        s_tokens = len(s.split())
        # Jeśli dodanie zdania przekracza target_tokens,
        # zapisz obecny chunk i zacznij nowy
        if current_tokens + s_tokens > target_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = s
            current_tokens = s_tokens
        else:
            current_chunk += " " + s
            current_tokens += s_tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def default_length_function(text):
    """
    Domyślna funkcja licząca tokeny – wykorzystuje podział po białych znakach.
    Możesz ją zastąpić np. funkcją openai_token_count, która dokładniej liczy tokeny.
    """
    return len(text.split())

###############################
# Klasa ClusterSemanticChunker
###############################

class ClusterSemanticChunker:
    def __init__(self, embedding_function, max_chunk_size, length_function=default_length_function,
                 base_fragment_target=15):
        """
        Args:
            embedding_function (callable): Funkcja przyjmująca listę tekstów i zwracająca tablicę embeddingów,
                                             np. z modelu typu SentenceTransformer.
            max_chunk_size (int): Maksymalna liczba tokenów w jednym chunku.
            length_function (callable): Funkcja licząca liczbę tokenów w tekście.
            base_fragment_target (int): Docelowa liczba tokenów przy rekurencyjnym podziale na fragmenty bazowe.
        """
        self.embedding_function = embedding_function
        self.max_chunk_size = max_chunk_size
        self.length_function = length_function
        self.base_fragment_target = base_fragment_target

    def split_text(self, document):
        """
        Główna metoda – przyjmuje dokument (tekst) i zwraca listę chunków.
        Proces:
          1. Rozbicie dokumentu na małe fragmenty bazowe.
          2. Uzyskanie embeddingów dla fragmentów.
          3. Budowanie macierzy podobieństwa.
          4. Dynamic programming – optymalny podział fragmentów na chunki zgodnie z ograniczeniem rozmiaru i
             maksymalizacją spójności semantycznej.
        """
        # 1. Rozbijamy dokument na fragmenty bazowe (ok. base_fragment_target tokenów)
        base_fragments = simple_recursive_split(document, target_tokens=self.base_fragment_target)
        n = len(base_fragments)
        if n == 0:
            return []

        # 2. Oblicz embeddingi – zakładamy, że embedding_function przyjmuje listę tekstów
        embeddings = self.embedding_function(base_fragments)
        # Sprawdźmy, że embeddings to macierz numpy
        embeddings = np.array(embeddings)
        
        # 3. Oblicz macierz podobieństwa między wszystkimi fragmentami
        sim_matrix = cosine_similarity(embeddings)  # kształt (n, n)
        
        # 4. Dynamic programming – optymalny podział fragmentów na chunki.
        # dp[i] = maksymalny reward (sumę podobieństw) dla segmentacji fragmentów i..n-1
        # next_split[i] = indeks, gdzie następuje podział po najlepszym chunku zaczynającym się od i
        dp = np.full(n+1, -np.inf)
        dp[n] = 0  # gdy nie ma już fragmentów
        next_split = [-1] * (n + 1)
        
        # Precompute token counts dla fragmentów
        token_counts = [self.length_function(fragment) for fragment in base_fragments]
        
        # Funkcja obliczająca reward dla chunka zawierającego fragmenty i...j-1
        # Reward: suma cosine similarity dla każdej pary fragmentów w chunce.
        def chunk_reward(i, j):
            if j - i < 2:
                return 0  # pojedynczy fragment – brak par
            reward = 0
            for p in range(i, j):
                for q in range(p+1, j):
                    reward += sim_matrix[p, q]
            return reward
        
        # Dynamic programming: iterujemy od końca do początku
        # Dla pozycji i, rozważamy możliwe j tak, aby suma tokenów z i do j-1 nie przekroczyła max_chunk_size
        for i in range(n-1, -1, -1):
            total_tokens = 0
            best_reward = -np.inf
            best_j = i + 1
            for j in range(i+1, n+1):
                total_tokens += token_counts[j-1]
                if total_tokens > self.max_chunk_size:
                    break
                current_reward = chunk_reward(i, j) + dp[j]
                if current_reward > best_reward:
                    best_reward = current_reward
                    best_j = j
            dp[i] = best_reward
            next_split[i] = best_j

        # Odtwarzamy optymalny podział
        chunks = []
        i = 0
        while i < n:
            j = next_split[i]
            # Łączymy fragmenty od i do j-1
            chunk_text = " ".join(base_fragments[i:j])
            chunks.append(chunk_text)
            i = j
        
        return chunks

##################################
# Przykładowe użycie
##################################

if __name__ == "__main__":
    # Sprawdzamy czy biblioteka jest zainstalowana, jeśli nie, instalujemy
    try:
        import sentence_transformers
    except ImportError:
        print("Instaluję bibliotekę sentence_transformers...")
        os.system("pip install -U sentence-transformers")
        from sentence_transformers import SentenceTransformer

    # Ładujemy model embeddingowy - paraphrase-multilingual-MiniLM-L12-v2 jest lepszym wyborem
    # dla języka polskiego, ponieważ jest to model wielojęzyczny trenowany na wielu językach
    print("Ładowanie modelu embeddingowego dla języka polskiego...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def embedding_function(text_list):
        return model.encode(text_list, show_progress_bar=True)

    # Wczytujemy dane z pliku w folderze input
    input_dir = "input"
    output_file = "segmented_output.txt"
    
    try:
        # Sprawdzamy czy folder input istnieje
        if not os.path.exists(input_dir):
            print(f"Folder {input_dir} nie istnieje. Utwórz folder input.")
            exit(1)
            
        # Pobieramy listę plików z folderu input
        input_files = os.listdir(input_dir)
        if not input_files:
            print(f"Brak plików w folderze {input_dir}.")
            exit(1)
            
        # Używamy pierwszego znalezionego pliku
        input_file = os.path.join(input_dir, input_files[0])
        
        with open(input_file, "r", encoding="utf-8") as f:
            document = f.read()
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        exit(1)
    
    print(f"Wczytano tekst z pliku {input_file} ({len(document)} znaków)")
    
    # Tworzymy instancję chunkera z limitem 500 tokenów
    cluster_chunker = ClusterSemanticChunker(
        embedding_function=embedding_function,
        max_chunk_size=180,
        length_function=default_length_function
    )

    # Dzielimy dokument na chunki
    print("Segmentacja tekstu w toku...")
    chunks = cluster_chunker.split_text(document)

    # Zapisujemy wyniki do pliku - tylko chunki oddzielone akapitami
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk)
            f.write("\n\n")  # Oddzielamy chunki tylko akapitem
    
    print(f"Utworzono {len(chunks)} segmentów. Wyniki zapisano do pliku {output_file}")
