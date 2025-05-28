import re
import os
import nltk
from sklearn.metrics import adjusted_rand_score
from collections import Counter

nltk.download('punkt', quiet=True)

def simple_sentence_tokenize(text):
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
    content = load_text_from_file(filename)
    if content is None:
        return []
    raw_segments = [seg.strip() for seg in content.split('\n\n') if seg.strip()]
    segments = []
    for seg in raw_segments:
        lines = seg.splitlines()
        if lines and lines[0].lower().startswith("segment"):
            segment_text = "\n".join(lines[1:]).strip()
        else:
            segment_text = "\n".join(lines).strip()
        if segment_text:
            segments.append(segment_text)
    return segments

def get_labels_from_segments(segments, sentences):
    labels = []
    idx = 0
    for seg_id, segment in enumerate(segments):
        seg_sentences = simple_sentence_tokenize(segment)
        num_sent = len(seg_sentences)
        for _ in range(num_sent):
            if idx < len(sentences):
                labels.append(seg_id)
                idx += 1
    while len(labels) < len(sentences):
        labels.append(seg_id)
    return labels

def print_segment_distribution(labels, method_name):
    counter = Counter(labels)
    print(f"\nRozkład zdań w segmentach ({method_name}):")
    for seg_id in sorted(counter):
        print(f"  Segment {seg_id}: {counter[seg_id]} zdań")

def main(input_filename, direct_filename, iterative_filename):
    if not os.path.isfile(input_filename):
        print(f"Plik {input_filename} nie istnieje.")
        return
    if not os.path.isfile(direct_filename):
        print(f"Plik {direct_filename} nie istnieje.")
        return
    if not os.path.isfile(iterative_filename):
        print(f"Plik {iterative_filename} nie istnieje.")
        return

    text = load_text_from_file(input_filename)
    sentences = simple_sentence_tokenize(text)
    print(f"Liczba zdań w oryginalnym tekście: {len(sentences)}")

    direct_segments = read_segments_from_file(direct_filename)
    iterative_segments = read_segments_from_file(iterative_filename)
    print(f"Liczba segmentów (direct): {len(direct_segments)}")
    print(f"Liczba segmentów (iterative): {len(iterative_segments)}")

    direct_labels = get_labels_from_segments(direct_segments, sentences)
    iterative_labels = get_labels_from_segments(iterative_segments, sentences)

    if len(direct_labels) != len(sentences) or len(iterative_labels) != len(sentences):
        print("Uwaga: Liczba etykiet nie odpowiada liczbie zdań. Sprawdź format plików segmentacji.")
        return

    ari_score = adjusted_rand_score(direct_labels, iterative_labels)
    print(f"\nAdjusted Rand Index (ARI) direct vs iterative: {ari_score:.4f}")

    print_segment_distribution(direct_labels, "direct")
    print_segment_distribution(iterative_labels, "iterative")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Plik z oryginalnym tekstem')
    parser.add_argument('--direct', required=True, help='Plik z segmentacją direct')
    parser.add_argument('--iterative', required=True, help='Plik z segmentacją iterative')
    args = parser.parse_args()
    main(args.input, args.direct, args.iterative)
