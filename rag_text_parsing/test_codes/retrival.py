import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Przygotowanie modelu do osadzania tekstu (embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Możesz wybrać inny model

# 2. Indeksowanie dokumentów
# Przykładowa kolekcja dokumentów (mogą to być notatki, artykuły, specyfikacje itp.)
documents = [
    "System sterowania oparty na algorytmie PID z wykorzystaniem czujników temperatury i ciśnienia.",
    "Nowoczesne procesory ARM stosowane w urządzeniach mobilnych oraz urządzeniach IoT.",
    "Zastosowanie robotyki w montażu, gdzie precyzyjne manipulatory wykonują zadania spawalnicze.",
    "Integracja systemów automatyki przemysłowej z rozwiązaniami IoT w celu optymalizacji procesów."
]

# Przekształć dokumenty na wektory
document_embeddings = embedding_model.encode(documents, convert_to_numpy=True)

# Utwórz indeks FAISS. W tym przykładzie używamy IndexFlatL2 (wyszukiwanie oparte na odległości L2).
embedding_dim = document_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(document_embeddings)  # Dodajemy wektory dokumentów do indeksu

# 3. Przekształcenie zapytania (kluczowych informacji)
# Załóżmy, że z Twojej ekstrakcji kluczowych informacji otrzymałeś poniższy tekst:
query_text = "Algorytm PID i czujniki w systemach automatyki przemysłowej"

# Przekształć zapytanie na wektor
query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)

# 4. Wyszukiwanie – pobranie najbliższych dokumentów
k = 3  # liczba dokumentów do pobrania
distances, indices = index.search(query_embedding, k)

# 5. Wyświetlenie wyników
print("Wyniki wyszukiwania (najbardziej podobne dokumenty):")
for idx in indices[0]:
    print(f"- {documents[idx]}")

