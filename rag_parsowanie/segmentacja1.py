import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from collections import defaultdict
import os

nltk.download('punkt', quiet=True)

def load_polish_stopwords(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            stopwords = [line.strip() for line in file if line.strip()]
        print(f"Wczytano {len(stopwords)} polskich stopwordów.")
        return stopwords
    except Exception as e:
        print(f"Błąd podczas wczytywania polskich stopwordów: {e}")
        return []

# Ścieżka do pliku z polskimi stopwordami
stopwords_filepath = 'polish_stopwords.txt'
polish_stop_words = load_polish_stopwords(stopwords_filepath)

def preprocess_text(text):
    """
    Poprawia formatowanie tekstu przed tokenizacją zdań:
    - łączy zdania rozdzielone na kilka akapitów
    - usuwa niepotrzebne znaki
    - poprawia formatowanie numeracji i odnośników
    """
    # Usunięcie znaczników filepath
    text = re.sub(r'//\s*filepath:.*?\n', '', text)
    
    # Łączenie linii, które zostały błędnie podzielone (zdania przerwane na końcu linii)
    text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)
    
    # Usunięcie numeracji rysunków i odnośników (Rys. X.X)
    text = re.sub(r'Rys\.\s+\d+\.\d+\.\s*', '', text)
    
    # Specjalne traktowanie numeracji rysunków - zaznaczenie jako oddzielne sekcje
    text = re.sub(r'(Rys\.\s+\d+\..*?)\n', r'\n\n\1\n\n', text)
    
    # Usunięcie dzielenia wyrazów z pomocą myślników i łączenie z następną linią
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Usunięcie numerów stron i oznaczeń strony
    text = re.sub(r'=== Strona \d+ ===', '', text)
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Łączenie paragrafów, które logicznie powinny być połączone
    # Jeśli akapit kończy się bez kropki, łączymy go z następnym
    text = re.sub(r'([^.!?:])\n\n([a-zęóąśłżźćń])', r'\1 \2', text)
    
    # Usunięcie wielokrotnych pustych linii
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def better_sentence_tokenize(text):
    """
    Lepszy tokenizator zdań uwzględniający specyfikę tekstu technicznego
    """
    # Najpierw wstępne przetwarzanie tekstu
    text = preprocess_text(text)
    
    # Zamiana wielu białych znaków na pojedyncze spacje
    text = re.sub(r'\s+', ' ', text)
    
    # Dzielenie na zdania używając standardowych znaków końca zdania
    # ale uważając na przypadki jak "rys. 4.5." które nie kończą zdania
    text = re.sub(r'(?<=[.!?])\s+(?=[A-ZŚĄĘŹŻŃŁÓĆ])', '\n', text)
    
    # Dzielimy na zdania
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    
    # Usuwanie zdań, które są tylko numeracją lub pojedynczymi literami
    sentences = [s for s in sentences if len(s) > 2 and not re.match(r'^\d+$', s)]
    
    return sentences

def detect_figure_captions(text):
    """Wykrywa podpisy pod rysunkami i oznacza je jako oddzielne segmenty"""
    caption_pattern = r'Rys\.\s+\d+\..*?(?=\n\n|\Z)'
    captions = re.finditer(caption_pattern, text, re.DOTALL)
    caption_positions = []
    
    for match in captions:
        start, end = match.span()
        caption_positions.append((start, end, match.group()))
    
    return caption_positions

def estimate_tokens(text):
    words = text.split()
    return len(words)

def split_long_segments(segments, max_tokens=500):
    result = []
    for segment in segments:
        if estimate_tokens(segment) <= max_tokens:
            result.append(segment)
            continue
        
        # Sprawdź czy segment to podpis pod rysunkiem - jeśli tak, nie dziel
        if re.match(r'^\s*Rys\.', segment):
            result.append(segment)
            continue
            
        sentences = better_sentence_tokenize(segment)
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            
            # Jeśli pojedyncze zdanie jest zbyt długie
            if sentence_tokens > max_tokens:
                if current_chunk:
                    result.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                    
                # Dzielenie długiego zdania na kawałki
                words = sentence.split()
                chunk = []
                chunk_tokens = 0
                for word in words:
                    if chunk_tokens + 1 > max_tokens:
                        result.append(" ".join(chunk))
                        chunk = [word]
                        chunk_tokens = 1
                    else:
                        chunk.append(word)
                        chunk_tokens += 1
                        
                if chunk:
                    result.append(" ".join(chunk))
            elif current_tokens + sentence_tokens > max_tokens:
                result.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                
        if current_chunk:
            result.append(" ".join(current_chunk))
            
    return result

def detect_section_boundaries(text):
    """Wykrywa granice sekcji na podstawie formatowania tekstu"""
    # Wykrywanie nagłówków (całe zdania z wielkich liter)
    headers = re.finditer(r'\n([A-ZŚĄĘŹŻŃŁÓĆ][A-ZŚĄĘŹŻŃŁÓĆ\s]+)(?:\n|$)', text)
    boundaries = []
    
    for match in headers:
        boundaries.append((match.start(), match.group(1)))
    
    # Wykrywanie pustych linii jako potencjalnych granic sekcji
    paragraphs = text.split('\n\n')
    position = 0
    
    for paragraph in paragraphs:
        if paragraph.strip():
            position += len(paragraph) + 2  # +2 for '\n\n'
            boundaries.append((position-2, None))  # -2 to position at the end of paragraph
    
    # Sortowanie granic po pozycji
    boundaries.sort(key=lambda x: x[0])
    
    return boundaries

def merge_similar_segments(segments, similarity_threshold=0.3):
    """Łączy podobne segmenty, aby uniknąć zbyt rozdrobnionego podziału"""
    if len(segments) <= 1:
        return segments
        
    vectorizer = TfidfVectorizer(stop_words=polish_stop_words)
    try:
        X = vectorizer.fit_transform(segments)
        
        merged_segments = []
        current_segment = segments[0]
        current_vector = X[0].toarray().flatten()
        
        for i in range(1, len(segments)):
            segment_vector = X[i].toarray().flatten()
            
            if np.sum(current_vector) > 0 and np.sum(segment_vector) > 0:
                similarity = np.dot(current_vector, segment_vector) / (
                    np.linalg.norm(current_vector) * np.linalg.norm(segment_vector)
                )
            else:
                similarity = 0
                
            # Jeśli segment jest podobny do poprzedniego i połączony nie przekroczy limitu
            if (similarity > similarity_threshold and 
                estimate_tokens(current_segment + " " + segments[i]) <= 500):
                current_segment += " " + segments[i]
                current_vector = 0.5 * current_vector + 0.5 * segment_vector
            else:
                merged_segments.append(current_segment)
                current_segment = segments[i]
                current_vector = segment_vector
                
        merged_segments.append(current_segment)
        return merged_segments
    except Exception as e:
        print(f"Błąd podczas łączenia podobnych segmentów: {e}")
        return segments

def improved_segmentation(text, max_tokens=500):
    """Ulepszona segmentacja tekstu z uwzględnieniem specyfiki tekstów technicznych"""
    # Wstępne przetwarzanie tekstu
    text = preprocess_text(text)
    
    # Próba wykrycia naturalnych paragrafów
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Wykrywanie podpisów pod rysunkami i oznaczeń sekcji
    captions = detect_figure_captions(text)
    
    # Jeśli mamy więcej niż jeden paragraf, użyj ich jako podstawowej segmentacji
    segments = []
    if len(paragraphs) > 1:
        print(f"Wykryto {len(paragraphs)} naturalnych paragrafów")
        
        # Podziel paragrafy które są za długie
        for paragraph in paragraphs:
            # Sprawdź czy paragraf to podpis rysunku - jeśli tak, dodaj jako osobny segment
            if re.match(r'^\s*Rys\.', paragraph):
                segments.append(paragraph)
            # Jeśli paragraf jest za długi, podziel go
            elif estimate_tokens(paragraph) > max_tokens:
                paragraph_sentences = better_sentence_tokenize(paragraph)
                sub_segments = []
                current_segment = []
                current_tokens = 0
                
                for sentence in paragraph_sentences:
                    sentence_tokens = estimate_tokens(sentence)
                    if current_tokens + sentence_tokens > max_tokens:
                        if current_segment:
                            sub_segments.append(" ".join(current_segment))
                        current_segment = [sentence]
                        current_tokens = sentence_tokens
                    else:
                        current_segment.append(sentence)
                        current_tokens += sentence_tokens
                
                if current_segment:
                    sub_segments.append(" ".join(current_segment))
                segments.extend(sub_segments)
            else:
                segments.append(paragraph)
    else:
        # Jeśli nie ma naturalnych paragrafów, użyj bardziej zaawansowanej segmentacji
        sentences = better_sentence_tokenize(text)
        
        # Używaj TF-IDF i clusteringu do wykrycia segmentów tematycznych
        if len(sentences) > 10:
            try:
                vectorizer = TfidfVectorizer(stop_words=polish_stop_words)
                X = vectorizer.fit_transform(sentences)
                
                # Określ liczbę klastrów na podstawie długości tekstu
                n_clusters = max(2, min(5, len(sentences) // 10))
                
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering.fit_predict(X.toarray())
                
                # Grupuj zdania według klastrów
                clusters = defaultdict(list)
                for i, label in enumerate(labels):
                    clusters[label].append(i)
                
                # Sortuj klastry według pozycji pierwszego zdania
                sorted_clusters = sorted(clusters.keys(), key=lambda k: min(clusters[k]))
                
                for cluster in sorted_clusters:
                    cluster_sentences = [sentences[i] for i in clusters[cluster]]
                    cluster_text = " ".join(cluster_sentences)
                    
                    # Jeśli segment jest za długi, podziel go dalej
                    if estimate_tokens(cluster_text) > max_tokens:
                        segments.extend(split_long_segments([cluster_text], max_tokens))
                    else:
                        segments.append(cluster_text)
            except Exception as e:
                print(f"Błąd podczas wykonywania zaawansowanej segmentacji: {e}")
                # W razie błędu spróbuj prostszej metody
                segments = split_long_segments([text], max_tokens)
        else:
            segments = split_long_segments([text], max_tokens)
    
    # Łączenie podobnych segmentów, aby uniknąć nadmiernej fragmentacji
    if len(segments) > 5:
        segments = merge_similar_segments(segments)
    
    # Usuwanie duplikatów i pustych segmentów
    segments = [s.strip() for s in segments if s.strip()]
    unique_segments = []
    for segment in segments:
        if segment not in unique_segments:
            unique_segments.append(segment)
    
    print(f"Finalna liczba segmentów: {len(unique_segments)}")
    return unique_segments

def save_segments_to_file(segments, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for i, segment in enumerate(segments, 1):
            token_count = estimate_tokens(segment)
            file.write(f"Segment {i} (około {token_count} tokenów):\n{segment}\n\n")

def load_text_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku {filename}: {e}")
        return None

def main():
    MAX_TOKENS = 500
    input_filename = 'processed_input.txt'
    if not os.path.isfile(input_filename):
        print(f"Plik {input_filename} nie istnieje. Upewnij się, że plik znajduje się w folderze roboczym.")
        print(f"Aktualny katalog roboczy: {os.getcwd()}")
        return
    text = load_text_from_file(input_filename)
    if not text:
        print("Nie udało się wczytać tekstu. Program zostanie zakończony.")
        return
    print(f"Wczytano tekst o długości {len(text)} znaków, około {estimate_tokens(text)} tokenów")
    
    print("Wykonywanie ulepszonej segmentacji...")
    improved_segments = improved_segmentation(text, MAX_TOKENS)
    
    improved_output = 'improved_segments.txt'
    save_segments_to_file(improved_segments, improved_output)
    
    print(f"Ulepszona segmentacja: utworzono {len(improved_segments)} segmentów")
    print(f"Wyniki zapisano do pliku '{improved_output}'")

if __name__ == "__main__":
    main()