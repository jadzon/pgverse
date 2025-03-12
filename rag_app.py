from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from transformers import pipeline
import numpy as np
import os

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

# Konfiguracja Neo4j
NEO4J_URI = "neo4j+s://73392314.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "3eoM3MrYCWblXJp4GOML9pwDFhJhyJo3x6C2JwiLkQ4"

def load_document(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def chunk_text(text, chunk_size=500, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)

def generate_embeddings(chunks, model_name='all-MiniLM-L12-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

#Tworzenie indeksu wektorowego w Neo4j
def create_vector_index(tx):
    tx.run("""
        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
        }}
    """)

#Zapisywanie chunków i embeddingów do Neo4j
def store_chunks(tx, chunks, embeddings):
    for chunk, embedding in zip(chunks, embeddings):
        tx.run("""
            CREATE (c:Chunk {text: $text, embedding: $embedding})
        """, text=chunk, embedding=embedding.tolist())

#Wyszukiwanie podobnych fragmentów
def find_similar_chunks(tx, query_embedding, top_k=5):
    result = tx.run("""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding) 
        YIELD node, score
        RETURN node.text AS text, score
    """, query_embedding=query_embedding.tolist(), top_k=top_k)
    
    return [{"text": record["text"], "score": record["score"]} for record in result]

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    
    prompt = f"""Na podstawie poniższego kontekstu odpowiedz na pytanie użytkownika.
Jeśli odpowiedź nie wynika jasno z kontekstu, napisz że nie masz wystarczających informacji.

Kontekst:
{context}

Pytanie:
{question}

Odpowiedź:"""

    #Inicjalizacja modelu BloomZ 
    generator = pipeline("text-generation", model="bigscience/bloomz-560m", device=0)  #GPU (device=0)

    response = generator(prompt, max_new_tokens=200)[0]['generated_text']
    
    answer_start = response.find("Odpowiedź:") + len("Odpowiedź:")
    answer = response[answer_start:].strip()
    
    return answer

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    corpus = load_document("kinematyka.txt")
    chunks = chunk_text(corpus)
    embeddings = generate_embeddings(chunks)

    with driver.session() as session:
        session.execute_write(create_vector_index)
        session.execute_write(store_chunks, chunks, embeddings)

    print("Dane zostały zapisane w bazie Neo4j.")

    embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

    while True:
        question = input("\n Wpisz swoje pytanie (lub wpisz 'exit' aby zakończyć): ")
        
        if question.lower() == 'exit':
            break
        
        query_embedding = embedding_model.encode(question)

        with driver.session() as session:
            similar_chunks_data = session.execute_read(find_similar_chunks, query_embedding)

        context_chunks = [chunk["text"] for chunk in similar_chunks_data]

        context_chunks = context_chunks[:3]

        answer = generate_answer(question, context_chunks)

        print(f"\nOdpowiedź LLM'a:\n{answer}")

    driver.close()

if __name__ == "__main__":
    main()
