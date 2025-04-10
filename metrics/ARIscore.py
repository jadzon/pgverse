import re
import os
import nltk
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict

nltk.download('punkt', quiet=True)

def simple_sentence_tokenize(text):
    """
    Dzieli tekst na zdania na podstawie interpunkcji.
    """
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def load_text_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku {filename}: {e}")
        return None

def read_segments_from_file(filename):
    """
    Wczytuje segmenty z pliku wygenerowanego przez wcześniejszy skrypt.
    Zakładamy, że segmenty oddzielone są pustymi liniami, a pierwszy wiersz każdego segmentu
    zawiera nagłówek, który pomijamy.
    """
    content = load_text_from_file(filename)
    if content is None:
        return []
    raw_segments = [seg.strip() for seg in content.split('\n\n') if seg.strip()]
    segments = []
    for seg in raw_segments:
        lines = seg.splitlines()
        # Pomijamy linię nagłówkową, jeśli zawiera słowo "Segment"
        if lines and lines[0].lower().startswith("segment"):
            segment_text = "\n".join(lines[1:]).strip()
        else:
            segment_text = "\n".join(lines).strip()
        if segment_text:
            segments.append(segment_text)
    return segments

def get_labels_from_segments(segments, sentences):
    """
    Dla listy segmentów przypisujemy każdemu zdaniu etykietę segmentu, do którego należy.
    Zakładamy, że segmenty są tworzone z kolejnych zdań oryginalnego tekstu.
    """
    labels = []
    idx = 0
    for seg_id, segment in enumerate(segments):
        seg_sentences = simple_sentence_tokenize(segment)
        num_sent = len(seg_sentences)
        for _ in range(num_sent):
            if idx < len(sentences):
                labels.append(seg_id)
                idx += 1
    # Jeśli nie przypisano etykiety dla wszystkich zdań, przypisujemy pozostałe do ostatniego segmentu.
    while len(labels) < len(sentences):
        labels.append(seg_id)
    return labels

def main():
    # Pliki wejściowe
    input_filename = 'input.txt'
    direct_filename = 'direct_output.txt'
    iterative_filename = 'iterative_output.txt'
    perfect_filename = 'perfect.txt'  # Plik z wzorcową segmentacją
    
    if not os.path.isfile(input_filename):
        print(f"Plik {input_filename} nie istnieje.")
        return
    if not os.path.isfile(direct_filename):
        print(f"Plik {direct_filename} nie istnieje.")
        return
    if not os.path.isfile(iterative_filename):
        print(f"Plik {iterative_filename} nie istnieje.")
        return
    if not os.path.isfile(perfect_filename):
        print(f"Plik {perfect_filename} nie istnieje.")
        print("Nie można porównać metod z wzorcową segmentacją.")
        perfect_segments = None
    else:
        perfect_segments = read_segments_from_file(perfect_filename)
        print(f"Wczytano wzorcową segmentację: {len(perfect_segments)} segmentów")

    # Wczytanie oryginalnego tekstu i podział na zdania
    text = load_text_from_file(input_filename)
    sentences = simple_sentence_tokenize(text)
    print(f"Liczba zdań w oryginalnym tekście: {len(sentences)}")
    
    # Wczytanie segmentacji z plików
    direct_segments = read_segments_from_file(direct_filename)
    iterative_segments = read_segments_from_file(iterative_filename)
    
    print(f"Liczba segmentów (metoda bezpośrednia): {len(direct_segments)}")
    print(f"Liczba segmentów (metoda iteracyjna): {len(iterative_segments)}")
    
    # Przypisanie etykiet do zdań dla obu metod
    direct_labels = get_labels_from_segments(direct_segments, sentences)
    iterative_labels = get_labels_from_segments(iterative_segments, sentences)
    
    # Jeśli liczba etykiet nie zgadza się z liczbą zdań, ostrzegamy użytkownika.
    if len(direct_labels) != len(sentences) or len(iterative_labels) != len(sentences):
        print("Uwaga: Liczba etykiet nie odpowiada liczbie zdań. Sprawdź format plików segmentacji.")
        return

    # Obliczenie ARI między metodami
    ari_score_between_methods = adjusted_rand_score(direct_labels, iterative_labels)
    print(f"Współczynnik Adjusted Rand Index (ARI) pomiędzy metodami: {ari_score_between_methods:.4f}")
    
    # Porównanie z wzorcową segmentacją, jeśli dostępna
    if perfect_segments:
        perfect_labels = get_labels_from_segments(perfect_segments, sentences)
        
        if len(perfect_labels) != len(sentences):
            print("Uwaga: Liczba etykiet wzorcowej segmentacji nie odpowiada liczbie zdań.")
        else:
            # Obliczenie ARI dla metody bezpośredniej względem wzorca
            ari_direct_vs_perfect = adjusted_rand_score(direct_labels, perfect_labels)
            print(f"ARI metody bezpośredniej względem wzorca: {ari_direct_vs_perfect:.4f}")
            
            # Obliczenie ARI dla metody iteracyjnej względem wzorca
            ari_iterative_vs_perfect = adjusted_rand_score(iterative_labels, perfect_labels)
            print(f"ARI metody iteracyjnej względem wzorca: {ari_iterative_vs_perfect:.4f}")
            
            # Określenie, która metoda jest lepsza
            if ari_direct_vs_perfect > ari_iterative_vs_perfect:
                print("Metoda bezpośrednia jest lepsza (bliższa wzorcowej segmentacji).")
                improvement = ((ari_direct_vs_perfect - ari_iterative_vs_perfect) / 
                              max(abs(ari_iterative_vs_perfect), 0.0001)) * 100
                print(f"Poprawa o {improvement:.2f}% względem metody iteracyjnej.")
            elif ari_iterative_vs_perfect > ari_direct_vs_perfect:
                print("Metoda iteracyjna jest lepsza (bliższa wzorcowej segmentacji).")
                improvement = ((ari_iterative_vs_perfect - ari_direct_vs_perfect) / 
                              max(abs(ari_direct_vs_perfect), 0.0001)) * 100
                print(f"Poprawa o {improvement:.2f}% względem metody bezpośredniej.")
            else:
                print("Obie metody są tak samo dobre (mają taki sam ARI względem wzorcowej segmentacji).")
    
    # Obliczamy statystyki dla obu metod
    direct_segments_count = len(set(direct_labels))
    iterative_segments_count = len(set(iterative_labels))
    
    print(f"Liczba unikalnych segmentów (metoda bezpośrednia): {direct_segments_count}")
    print(f"Liczba unikalnych segmentów (metoda iteracyjna): {iterative_segments_count}")
    
    # Obliczamy rozkład zdań w segmentach
    direct_segment_sizes = {}
    iterative_segment_sizes = {}
    
    for label in direct_labels:
        direct_segment_sizes[label] = direct_segment_sizes.get(label, 0) + 1
    
    for label in iterative_labels:
        iterative_segment_sizes[label] = iterative_segment_sizes.get(label, 0) + 1
    
    print("\nRozkład zdań w segmentach (metoda bezpośrednia):")
    for seg_id, count in sorted(direct_segment_sizes.items()):
        print(f"  Segment {seg_id}: {count} zdań")
    
    print("\nRozkład zdań w segmentach (metoda iteracyjna):")
    for seg_id, count in sorted(iterative_segment_sizes.items()):
        print(f"  Segment {seg_id}: {count} zdań")
    
    if perfect_segments:
        perfect_segment_sizes = {}
        for label in perfect_labels:
            perfect_segment_sizes[label] = perfect_segment_sizes.get(label, 0) + 1
        
        print("\nRozkład zdań w segmentach (wzorcowa segmentacja):")
        for seg_id, count in sorted(perfect_segment_sizes.items()):
            print(f"  Segment {seg_id}: {count} zdań")

if __name__ == "__main__":
    main()