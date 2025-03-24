import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from collections import defaultdict
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def simple_sentence_tokenize(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def estimate_tokens(text):
    words = text.split()
    return len(words)

def split_long_segments(segments, max_tokens=500):
    result = []
    for segment in segments:
        if estimate_tokens(segment) <= max_tokens:
            result.append(segment)
            continue
        sentences = simple_sentence_tokenize(segment)
        current_chunk = []
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            if sentence_tokens > max_tokens:
                if current_chunk:
                    result.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
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

def detect_topic_changes(sentences):
    if len(sentences) <= 2:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9, ngram_range=(1, 2))
    sentence_vectors = vectorizer.fit_transform(sentences)
    n_clusters = max(2, min(5, len(sentences) // 3))
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    labels = clustering.fit_predict(sentence_vectors.toarray())
    boundaries = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            boundaries.append(i)
    return boundaries

def detect_topic_changes_alternative(sentences):
    if len(sentences) <= 2:
        return []
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    similarities = []
    for i in range(len(sentences) - 1):
        vec1 = X[i].toarray().flatten()
        vec2 = X[i+1].toarray().flatten()
        if np.sum(vec1) > 0 and np.sum(vec2) > 0:
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(sim)
        else:
            similarities.append(0)
    boundaries = []
    for i in range(1, len(similarities) - 1):
        if (similarities[i] < similarities[i-1] and similarities[i] < similarities[i+1] and similarities[i] < 0.3):
            boundaries.append(i + 1)
    return boundaries

def direct_segmentation(text, max_tokens=500):
    sentences = simple_sentence_tokenize(text)
    if len(sentences) <= 2:
        if estimate_tokens(text) > max_tokens:
            return split_long_segments([text], max_tokens)
        return [text]
    try:
        boundaries = detect_topic_changes(sentences)
    except Exception as e:
        print(f"Błąd podczas wykrywania zmian tematu: {e}")
        boundaries = detect_topic_changes_alternative(sentences)
    if not boundaries:
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(sentences)
            n_clusters = max(2, min(4, len(sentences) // 4))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            segments_by_cluster = defaultdict(list)
            for i, label in enumerate(labels):
                segments_by_cluster[label].append(i)
            sorted_clusters = sorted(segments_by_cluster.keys(), key=lambda k: min(segments_by_cluster[k]))
            segments = []
            for cluster in sorted_clusters:
                segment_indices = segments_by_cluster[cluster]
                segment_sentences = [sentences[i] for i in segment_indices]
                segments.append(" ".join(segment_sentences))
            print(f"Wykryto {len(segments)} segmentów tematycznych (metoda K-means)")
            segments = split_long_segments(segments, max_tokens)
            print(f"Po podziale długich segmentów: {len(segments)} segmentów")
            return segments
        except Exception as e:
            print(f"Błąd podczas segmentacji alternatywnej: {e}")
            return split_long_segments([text], max_tokens)
    segments = []
    start = 0
    for boundary in boundaries:
        if boundary > start:
            segments.append(" ".join(sentences[start:boundary]))
            start = boundary
    if start < len(sentences):
        segments.append(" ".join(sentences[start:]))
    print(f"Wykryto {len(segments)} segmentów tematycznych (metoda bezpośrednia)")
    segments = split_long_segments(segments, max_tokens)
    print(f"Po podziale długich segmentów: {len(segments)} segmentów")
    return segments

def iterative_segmentation(text, max_tokens=500):
    sentences = simple_sentence_tokenize(text)
    if len(sentences) <= 2:
        if estimate_tokens(text) > max_tokens:
            return split_long_segments([text], max_tokens)
        return [text]
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
    X = vectorizer.fit_transform(sentences)
    similarities = []
    for i in range(len(sentences) - 1):
        vec1 = X[i].toarray().flatten()
        vec2 = X[i+1].toarray().flatten()
        if np.sum(vec1) > 0 and np.sum(vec2) > 0:
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(sim)
        else:
            similarities.append(0)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) > 1:
        print(f"Wykryto {len(paragraphs)} naturalnych paragrafów")
        paragraphs = split_long_segments(paragraphs, max_tokens)
        print(f"Po podziale długich paragrafów: {len(paragraphs)} segmentów")
        return paragraphs
    if similarities:
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        similarity_threshold = max(0.2, mean_sim - 0.5 * std_sim)
        print(f"Adaptacyjny próg podobieństwa: {similarity_threshold:.4f}")
    else:
        similarity_threshold = 0.3
    segments = []
    current_segment = [sentences[0]]
    current_topic_vector = X[0].toarray().flatten()
    current_tokens = estimate_tokens(sentences[0])
    for i in range(1, len(sentences)):
        sentence_vector = X[i].toarray().flatten()
        sentence_tokens = estimate_tokens(sentences[i])
        if current_tokens + sentence_tokens > max_tokens:
            segments.append(" ".join(current_segment))
            current_segment = [sentences[i]]
            current_topic_vector = sentence_vector
            current_tokens = sentence_tokens
            continue
        if np.sum(current_topic_vector) > 0 and np.sum(sentence_vector) > 0:
            similarity = np.dot(current_topic_vector, sentence_vector) / (
                np.linalg.norm(current_topic_vector) * np.linalg.norm(sentence_vector)
            )
        else:
            similarity = 0
        if similarity < similarity_threshold:
            if len(current_segment) >= 2 or len(segments) == 0:
                segments.append(" ".join(current_segment))
                current_segment = [sentences[i]]
                current_topic_vector = sentence_vector
                current_tokens = sentence_tokens
            else:
                current_segment.append(sentences[i])
                current_topic_vector = 0.4 * current_topic_vector + 0.6 * sentence_vector
                current_tokens += sentence_tokens
        else:
            current_segment.append(sentences[i])
            current_topic_vector = 0.7 * current_topic_vector + 0.3 * sentence_vector
            current_tokens += sentence_tokens
    if current_segment:
        segments.append(" ".join(current_segment))
    if len(segments) > len(sentences) // 3:
        print("Zbyt wiele segmentów, używam podejścia opartego na paragrafach")
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            paragraphs = split_long_segments(paragraphs, max_tokens)
            return paragraphs
    print(f"Wykryto {len(segments)} segmentów tematycznych (metoda iteracyjna)")
    segments = split_long_segments(segments, max_tokens)
    print(f"Po podziale długich segmentów: {len(segments)} segmentów")
    return segments

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
    input_filename = 'input.txt'
    if not os.path.isfile(input_filename):
        print(f"Plik {input_filename} nie istnieje. Upewnij się, że plik znajduje się w folderze roboczym.")
        print(f"Aktualny katalog roboczy: {os.getcwd()}")
        return
    text = load_text_from_file(input_filename)
    if not text:
        print("Nie udało się wczytać tekstu. Program zostanie zakończony.")
        return
    print(f"Wczytano tekst o długości {len(text)} znaków, około {estimate_tokens(text)} tokenów")
    print("Wykonywanie segmentacji bezpośredniej...")
    direct_segments = direct_segmentation(text, MAX_TOKENS)
    print("Wykonywanie segmentacji iteracyjnej...")
    iterative_segments = iterative_segmentation(text, MAX_TOKENS)
    direct_output = 'direct_output.txt'
    iterative_output = 'iterative_output.txt'
    save_segments_to_file(direct_segments, direct_output)
    save_segments_to_file(iterative_segments, iterative_output)
    print(f"Segmentacja bezpośrednia: utworzono {len(direct_segments)} segmentów")
    print(f"Segmentacja iteracyjna: utworzono {len(iterative_segments)} segmentów")
    print(f"Wyniki zapisano do plików '{direct_output}' i '{iterative_output}'")

if __name__ == "__main__":
    main()