import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Załaduj model spaCy do dzielenia na zdania
nlp = spacy.load("pl_core_news_md")

# Załaduj model do embeddingów zdań
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

def extract_focus_sentences(text, focus_instruction, similarity_threshold=0.5):
    """
    Funkcja zwraca zdania z tekstu, które mają cosine similarity z
    embeddingiem instrukcji wyższą lub równą similarity_threshold.
    
    Jeśli żadne zdanie nie spełni warunku, zwraca zdanie z najwyższym podobieństwem.
    """
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []

    # Tworzymy embeddingi: najpierw dla zdań, potem dla instrukcji
    sentence_embeddings = embedder.encode(sentences)
    instruction_embedding = embedder.encode([focus_instruction])
    
    # Obliczamy podobieństwo każdego zdania do instrukcji
    similarities = cosine_similarity(sentence_embeddings, instruction_embedding).flatten()
    
    # Wybieramy zdania przekraczające ustalony próg
    selected_sentences = [sent for sent, sim in zip(sentences, similarities) if sim >= similarity_threshold]
    
    # Jeśli żadne zdanie nie przekracza progu, zwracamy zdanie z najwyższym podobieństwem
    if not selected_sentences:
        max_index = np.argmax(similarities)
        selected_sentences = [sentences[max_index]]
    
    return selected_sentences

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input/processed_input.txt", help="Plik z tekstem źródłowym")
    parser.add_argument("--focus", default="Powody awarii silników", help="Instrukcja do ekstrakcji")
    parser.add_argument("--threshold", type=float, default=0.70, help="Próg podobieństwa (od 0 do 1)")
    parser.add_argument("--output", default="output_ex.txt", help="Nazwa pliku wyjściowego")
    args = parser.parse_args()

    # Wczytaj tekst z pliku wejściowego
    with open(args.input, "r", encoding="utf-8") as f:
        tekst = f.read()

    # Wywołaj funkcję ekstrakcji
    wynik = extract_focus_sentences(tekst, args.focus, similarity_threshold=args.threshold)

    # Ustal ścieżkę do folderu "output" i pliku wyjściowego
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)  # Utwórz folder, jeśli nie istnieje
    output_path = os.path.join(output_dir, args.output)

    # Zapisz wyniki do pliku w folderze "output"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"📌 Zdania spełniające próg podobieństwa ({args.threshold}) dla instrukcji: '{args.focus}'\n\n")
        for zdanie in wynik:
            f.write(f"• {zdanie}\n")
    
    print(f"\nWyniki zapisano do pliku: {output_path}")
    print(f"Znaleziono {len(wynik)} zdań spełniających kryterium podobieństwa.")