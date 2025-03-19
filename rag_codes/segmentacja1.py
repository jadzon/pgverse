import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Pobierz wymagane zasoby NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Funkcja do ręcznego podziału tekstu na zdania
def simple_sentence_tokenize(text):
    """
    Prosta funkcja do podziału tekstu na zdania używając wyrażeń regularnych.
    """
    # Usuń znaki nowej linii i normalizuj spacje
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Podziel tekst na zdania
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# Nowa funkcja do wykrywania punktów zmiany tematu
def detect_topic_changes(sentences):
    """
    Wykrywa granice tematyczne w tekście używając hierarchicznego klastrowania.
    """
    if len(sentences) <= 2:
        return []
    
    # Przekształć zdania na wektory TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2)
    )
    sentence_vectors = vectorizer.fit_transform(sentences)
    
    # Zastosuj hierarchiczne klastrowanie aglomeracyjne
    n_clusters = max(2, min(5, len(sentences) // 3))  # Inteligentna heurystyka dla liczby tematów
    
    # Używamy podstawowych parametrów, które są dostępne w AgglomerativeClustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average'  # Używamy average linkage dla stabilności
    )
    
    # Przekształć macierz rzadką na gęstą dla klastrowania
    labels = clustering.fit_predict(sentence_vectors.toarray())
    
    # Znajdź granice między klastrami (gdzie zmienia się temat)
    boundaries = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            boundaries.append(i)
    
    return boundaries

# Alternatywna implementacja wykrywania zmian tematów przy użyciu podobieństwa zdań
def detect_topic_changes_alternative(sentences):
    """
    Wykrywa granice tematyczne analizując spadki podobieństwa między sąsiednimi zdaniami.
    """
    if len(sentences) <= 2:
        return []
    
    # Wektoryzacja zdań
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    
    # Oblicz podobieństwo między sąsiednimi zdaniami
    similarities = []
    for i in range(len(sentences) - 1):
        vec1 = X[i].toarray().flatten()
        vec2 = X[i+1].toarray().flatten()
        
        if np.sum(vec1) > 0 and np.sum(vec2) > 0:
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(sim)
        else:
            similarities.append(0)
    
    # Znajdź lokalne minima (punkty, gdzie podobieństwo spada)
    boundaries = []
    for i in range(1, len(similarities) - 1):
        # Jeśli wartość jest mniejsza niż sąsiedzi i poniżej pewnego progu
        if (similarities[i] < similarities[i-1] and 
            similarities[i] < similarities[i+1] and 
            similarities[i] < 0.3):  # Można dostosować próg
            boundaries.append(i + 1)  # +1 bo i to indeks podobieństwa między zdaniami i i i+1
    
    return boundaries

# Funkcja do segmentacji bezpośredniej
def direct_segmentation(text):
    """
    Segmentuje tekst bezpośrednio na podstawie analizy tematycznej.
    """
    # Podziel tekst na zdania
    sentences = simple_sentence_tokenize(text)
    
    # Dla bardzo krótkiego tekstu, zwróć go jako jeden segment
    if len(sentences) <= 2:
        return [text]
    
    # Wykryj granice tematyczne
    try:
        boundaries = detect_topic_changes(sentences)
    except Exception as e:
        print(f"Błąd podczas wykrywania zmian tematu: {e}")
        # Spróbuj alternatywnej metody
        boundaries = detect_topic_changes_alternative(sentences)
    
    # Jeśli nie znaleziono granic, spróbuj z inną metodą
    if not boundaries:
        try:
            # Zastosuj K-means jako alternatywę
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(sentences)
            
            n_clusters = max(2, min(4, len(sentences) // 4))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Grupuj zdania według klastrów, zachowując ich kolejność
            segments_by_cluster = defaultdict(list)
            for i, label in enumerate(labels):
                segments_by_cluster[label].append(i)
            
            # Sortuj klastry według ich pierwszego wystąpienia
            sorted_clusters = sorted(segments_by_cluster.keys(), 
                                    key=lambda k: min(segments_by_cluster[k]))
            
            # Utwórz segmenty
            segments = []
            for cluster in sorted_clusters:
                segment_indices = segments_by_cluster[cluster]
                segment_sentences = [sentences[i] for i in segment_indices]
                segments.append(" ".join(segment_sentences))
            
            print(f"Wykryto {len(segments)} segmentów tematycznych (metoda K-means)")
            return segments
            
        except Exception as e:
            print(f"Błąd podczas segmentacji alternatywnej: {e}")
            return [text]
    
    # Utwórz segmenty na podstawie granic tematycznych
    segments = []
    start = 0
    
    for boundary in boundaries:
        if boundary > start:  # Upewnij się, że segment ma długość > 0
            segments.append(" ".join(sentences[start:boundary]))
            start = boundary
    
    # Dodaj ostatni segment
    if start < len(sentences):
        segments.append(" ".join(sentences[start:]))
    
    print(f"Wykryto {len(segments)} segmentów tematycznych (metoda bezpośrednia)")
    return segments

# Ulepszona funkcja do segmentacji iteracyjnej
def iterative_segmentation(text):
    """
    Segmentuje tekst iteracyjnie, analizując zmiany tematyczne z adaptacyjnym progiem.
    """
    # Podziel tekst na zdania
    sentences = simple_sentence_tokenize(text)
    
    # Dla bardzo krótkiego tekstu, zwróć go jako jeden segment
    if len(sentences) <= 2:
        return [text]
    
    # Inicjalizacja wektoryzatora TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=1, 
        max_df=0.9
    )
    
    # Wstępnie dopasuj wektoryzator do całego zestawu zdań
    X = vectorizer.fit_transform(sentences)
    
    # Oblicz podobieństwa między sąsiednimi zdaniami
    similarities = []
    for i in range(len(sentences) - 1):
        vec1 = X[i].toarray().flatten()
        vec2 = X[i+1].toarray().flatten()
        
        if np.sum(vec1) > 0 and np.sum(vec2) > 0:
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(sim)
        else:
            similarities.append(0)
    
    # Wykryj naturalne punkty podziału - poszukaj "przerw" w tekście
    # Możemy użyć podejścia bazującego na paragrafach w oryginalnym tekście
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Jeśli mamy naturalne paragrafy, użyjmy ich jako segmenty
    if len(paragraphs) > 1:
        print(f"Wykryto {len(paragraphs)} naturalnych paragrafów")
        return paragraphs
    
    # Ustaw adaptacyjny próg - nie za niski, żeby nie było zbyt wielu segmentów
    if similarities:
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        # Próg oparty na statystykach podobieństwa
        similarity_threshold = max(0.2, mean_sim - 0.5 * std_sim) 
        print(f"Adaptacyjny próg podobieństwa: {similarity_threshold:.4f}")
    else:
        similarity_threshold = 0.3  # Wartość domyślna, wyższa dla mniejszej liczby segmentów
    
    # Inicjalizuj segmenty i aktualny segment
    segments = []
    current_segment = [sentences[0]]
    current_topic_vector = X[0].toarray().flatten()
    
    # Iteruj przez pozostałe zdania
    for i in range(1, len(sentences)):
        # Pozyskaj wektor bieżącego zdania
        sentence_vector = X[i].toarray().flatten()
        
        # Oblicz podobieństwo między bieżącym zdaniem a aktualnym segmentem
        if np.sum(current_topic_vector) > 0 and np.sum(sentence_vector) > 0:
            similarity = np.dot(current_topic_vector, sentence_vector) / (
                np.linalg.norm(current_topic_vector) * np.linalg.norm(sentence_vector)
            )
        else:
            similarity = 0
        
        # Jeśli podobieństwo jest znacząco poniżej progu, zacznij nowy segment
        if similarity < similarity_threshold:
            # Dodatkowa weryfikacja: sprawdź czy segment ma rozsądną długość
            if len(current_segment) >= 2 or len(segments) == 0:
                # Zakończ bieżący segment
                segments.append(" ".join(current_segment))
                
                # Rozpocznij nowy segment
                current_segment = [sentences[i]]
                current_topic_vector = sentence_vector
            else:
                # Jeśli segment byłby zbyt krótki, kontynuuj bieżący
                current_segment.append(sentences[i])
                # Aktualizuj wektor tematu (większa waga dla nowego zdania)
                current_topic_vector = 0.4 * current_topic_vector + 0.6 * sentence_vector
        else:
            # Kontynuuj bieżący segment
            current_segment.append(sentences[i])
            
            # Aktualizuj wektor tematu (średnia ważona)
            current_topic_vector = 0.7 * current_topic_vector + 0.3 * sentence_vector
    
    # Dodaj ostatni segment
    if current_segment:
        segments.append(" ".join(current_segment))
    
    # Jeśli wyszło zbyt dużo segmentów, spróbuj je połączyć
    if len(segments) > len(sentences) // 3:
        print("Zbyt wiele segmentów, używam podejścia opartego na paragrafach")
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs
    
    print(f"Wykryto {len(segments)} segmentów tematycznych (metoda iteracyjna)")
    return segments

# Funkcja do zapisywania segmentów do pliku
def save_segments_to_file(segments, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for i, segment in enumerate(segments, 1):
            file.write(f"Segment {i}:\n{segment}\n\n")

# Przykładowy tekst do segmentacji
example_text = """
Sztuczna inteligencja coraz częściej wspomaga lekarzy w diagnozowaniu chorób. Algorytmy uczą się na podstawie tysięcy przypadków i potrafią wykryć nowotwory szybciej niż człowiek. Wciąż jednak pozostaje pytanie, czy pacjenci zaufają maszynom bardziej niż lekarzom. Tymczasem w świecie kosmosu astronomowie odkryli nową egzoplanetę w strefie nadającej się do zamieszkania. Wstępne analizy sugerują, że może tam istnieć woda w stanie ciekłym. To odkrycie budzi pytania o możliwość istnienia życia poza Ziemią. Równocześnie w branży technologicznej pojawił się nowy rodzaj klawiatury, który zmienia sposób interakcji z komputerem. Dzięki specjalnym czujnikom dostosowuje się do siły nacisku użytkownika, co może zrewolucjonizować ergonomię pracy biurowej. W edukacji natomiast rozwija się trend nauczania wspomaganego sztuczną inteligencją. Personalizowane lekcje dostosowane do tempa ucznia sprawiają, że tradycyjne metody nauczania mogą wkrótce stać się przestarzałe.
"""

# Główna funkcja programu
def main():
    print("Wykonywanie segmentacji bezpośredniej...")
    direct_segments = direct_segmentation(example_text)
    
    print("Wykonywanie segmentacji iteracyjnej...")
    iterative_segments = iterative_segmentation(example_text)
    
    # Zapisz wyniki do plików
    save_segments_to_file(direct_segments, "segmentacja_bezposrednia.txt")
    save_segments_to_file(iterative_segments, "segmentacja_iteracyjna.txt")
    
    print(f"Segmentacja bezpośrednia: utworzono {len(direct_segments)} segmentów")
    print(f"Segmentacja iteracyjna: utworzono {len(iterative_segments)} segmentów")
    print("Wyniki zapisano do plików 'segmentacja_bezposrednia.txt' i 'segmentacja_iteracyjna.txt'")

if __name__ == "__main__":
    main()