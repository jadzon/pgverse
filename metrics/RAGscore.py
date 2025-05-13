from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_alignment_score(generated_answer: str, retrieved_docs: list, model) -> float:
    """
    Oblicza Retrieval-Generation Alignment Score, czyli średnią cosine similarity pomiędzy
    embeddingiem wygenerowanej odpowiedzi a embeddingami pobranych dokumentów.

    Args:
        generated_answer (str): Tekst wygenerowanej odpowiedzi.
        retrieved_docs (list): Lista tekstów pobranych dokumentów (fragmentów).
        model: Wcześniej załadowany model SentenceTransformer.

    Returns:
        float: Średni wskaźnik podobieństwa (w skali 0-1).
    """
    # Obliczenie embeddingu dla wygenerowanej odpowiedzi
    gen_embedding = model.encode([generated_answer])
    
    # Obliczenie embeddingów dla pobranych fragmentów dokumentów
    docs_embeddings = model.encode(retrieved_docs)
    
    # Wyliczenie cosine similarity między odpowiedzią a każdym dokumentem
    similarities = cosine_similarity(gen_embedding, docs_embeddings)[0]
    
    # Średnie podobieństwo
    avg_similarity = np.mean(similarities)
    return avg_similarity

def main():
    # Wczytanie danych: wygenerowana odpowiedź i pobrane dokumenty
    try:
        with open("generated.txt", "r", encoding="utf-8") as f:
            generated_answer = f.read().strip()
    except FileNotFoundError:
        print("Plik 'generated.txt' nie został znaleziony.")
        return

    try:
        with open("retrieved.txt", "r", encoding="utf-8") as f:
            # Zakładamy, że każdy wiersz to jeden fragment dokumentu
            retrieved_docs = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print("Plik 'retrieved.txt' nie został znaleziony.")
        return

    # Załadowanie modelu SentenceTransformer – tutaj używamy modelu "all-MiniLM-L6-v2"
    print("Ładowanie modelu...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Obliczenie Retrieval-Generation Alignment Score
    alignment_score = compute_alignment_score(generated_answer, retrieved_docs, model)
    print(f"Retrieval-Generation Alignment Score: {alignment_score:.4f}")

    # Weryfikacja, czy wynik spełnia założenie (oczekiwany wynik > 0.60)
    if alignment_score > 0.60:
        print("Wynik przewidywany > 60% - Alignment Score spełnia wymagania.")
    else:
        print("Wynik poniżej 60% - warto zweryfikować jakość alignmentu.")

if __name__ == "__main__":
    main()
