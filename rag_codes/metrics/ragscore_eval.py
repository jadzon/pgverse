from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_alignment_score(generated_answer: str, retrieved_docs: list, model) -> float:
    gen_embedding = model.encode([generated_answer])
    docs_embeddings = model.encode(retrieved_docs)
    similarities = cosine_similarity(gen_embedding, docs_embeddings)[0]
    avg_similarity = np.mean(similarities)
    return avg_similarity

def main(generated_file='generated.txt', retrieved_file='retrieved.txt', model_name='all-MiniLM-L6-v2', threshold=0.60):
    try:
        with open(generated_file, "r", encoding="utf-8") as f:
            generated_answer = f.read().strip()
    except FileNotFoundError:
        print(f"Plik '{generated_file}' nie został znaleziony.")
        return
    try:
        with open(retrieved_file, "r", encoding="utf-8") as f:
            retrieved_docs = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Plik '{retrieved_file}' nie został znaleziony.")
        return
    print(f"Ładowanie modelu {model_name}...")
    model = SentenceTransformer(model_name)
    alignment_score = compute_alignment_score(generated_answer, retrieved_docs, model)
    print(f"Retrieval-Generation Alignment Score: {alignment_score:.4f}")
    if alignment_score > threshold:
        print(f"Wynik przewidywany > {int(threshold*100)}% - Alignment Score spełnia wymagania.")
    else:
        print(f"Wynik poniżej {int(threshold*100)}% - warto zweryfikować jakość alignmentu.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', default='generated.txt')
    parser.add_argument('--retrieved', default='retrieved.txt')
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--threshold', type=float, default=0.60)
    args = parser.parse_args()
    main(args.generated, args.retrieved, args.model, args.threshold)
