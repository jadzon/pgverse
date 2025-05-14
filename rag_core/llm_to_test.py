import os
import sys
from neo4j import GraphDatabase
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cohere
import threading
import time
import argparse
import torch

# Import additional libraries for Bielik model
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
except ImportError:
    print("Transformers library not found. Install it using: pip install transformers")

from rag_functions.embedding_chunker import CohereTextChunker
from rag_functions.embeddings import TextEmbedder
from rag_functions.graph import GraphBuilder, HybridTextRetriever, GraphPruner

def save_chunks_to_neo4j(file_path, api_key, neo4j_uri, neo4j_user, neo4j_password, max_tokens=500):
    """
    Chunks text, creates embeddings, and saves to Neo4j
    
    Args:
        file_path: Path to the text file
        api_key: Cohere API key
        neo4j_uri: URI for Neo4j database
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        max_tokens: Maximum tokens per chunk
    """
    # Initialize chunker and embedder
    chunker = CohereTextChunker(api_key)
    embedder = TextEmbedder(api_key)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    # Read and chunk the text
    print(f"Chunking text from {file_path}...")
    chunks = chunker.chunk_from_file(file_path, max_tokens)
    print(f"Created {len(chunks)} chunks")
    
    # Create and save embeddings for each chunk
    print("Generating embeddings and saving to Neo4j...")
    with driver.session() as session:
        # Create constraint for TextChunk nodes if it doesn't exist
        session.run("CREATE CONSTRAINT TextChunk_id IF NOT EXISTS FOR (n:TextChunk) REQUIRE n.id IS UNIQUE")
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = embedder.get_text_embedding(chunk).tolist()
            
            # Create a unique ID for the chunk
            chunk_id = f"chunk_{os.path.basename(file_path)}_{i}"
            
            # Save to Neo4j
            session.run("""
                MERGE (c:TextChunk {id: $id})
                SET c.text = $text,
                    c.embedding = $embedding,
                    c.source = $source,
                    c.timestamp = timestamp()
                """,
                id=chunk_id,
                text=chunk,
                embedding=embedding,
                source=file_path
            )
            
            if (i+1) % 10 == 0:
                print(f"Saved {i+1}/{len(chunks)} chunks")
    
    # Build relationships between chunks
    print("Creating relationships between similar chunks...")
    graph_builder = GraphBuilder(neo4j_uri, neo4j_user, neo4j_password, similarity_threshold=0.7)
    graph_builder.create_relationships()
    graph_builder.close()
    
    driver.close()
    print("Processing complete!")

def rag_query_cohere(query, api_key, neo4j_uri, neo4j_user, neo4j_password, top_k=3):
    """
    Wykonuje retrieval z Neo4j na podstawie embeddingu zapytania
    i generuje odpowiedź za pomocą Cohere Chat.
    """
    co = cohere.Client(api_key)
    # 1. Embed zapytanie używając TextEmbedder, żeby mieć tę samą liczbę wymiarów
    embedder = TextEmbedder(api_key)
    query_emb = embedder.get_text_embedding(query).tolist()

    # 2. Pobierz top_k najbardziej podobnych chunków z Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        result = session.run("""
            MATCH (c:TextChunk)
            WITH c, vector.similarity.cosine(c.embedding, $query_emb) AS score
            RETURN c.text AS text, score
            ORDER BY score DESC
            LIMIT $top_k
        """, query_emb=query_emb, top_k=top_k)
        contexts = [record["text"] for record in result]
    driver.close()

    # 3. Zbuduj prompt i wywołaj LLM - z ograniczonym kontekstem
    limited_context = []
    total_chars = 0
    char_limit = 2000  # Przybliżony limit znaków (ok. 2000 tokenów)
    
    for ctx in contexts:
        # Dodaj tylko tyle kontekstu, ile się zmieści
        if total_chars + len(ctx) <= char_limit:
            limited_context.append(ctx)
            total_chars += len(ctx)
        else:
            # Dodaj tylko część, jeśli cały się nie zmieści
            remaining = char_limit - total_chars
            if remaining > 100:  # Dodaj tylko jeśli zostało wystarczająco dużo miejsca
                limited_context.append(ctx[:remaining] + "...")
            break
    
    context_text = "\n\n".join(limited_context)
    print(f"Użyto {len(context_text)} znaków kontekstu")
    
    chat_resp = co.chat(
        message=f"Pytanie: {query}",
        model="command",
        preamble=f"Jesteś pomocnym asystentem, który odpowiada tylko na podstawie dostarczonego kontekstu. Używaj tylko poniższego kontekstu do odpowiedzi w języku polskim:\n\n{context_text}"
    )
    
    return chat_resp.text, contexts

def rag_query_bielik(query, neo4j_uri, neo4j_user, neo4j_password, model_name="speakleash/Bielik-11B-v2.3-Instruct", top_k=3):
    """
    Wykonuje retrieval z Neo4j i generuje odpowiedź za pomocą modelu Bielik.
    """
    try:
        # 1. Wczytaj model i tokenizer
        print("Ładowanie modelu Bielik...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        )
        streamer = TextStreamer(tokenizer)
        print("Model załadowany")
        
        # 2. Pobierz podobne chunki z Neo4j
        # Ponieważ Bielik nie ma własnych embedingów, używamy Cohere API
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Znajdź chunki za pomocą hybrydowego wyszukiwania
        hybrid = HybridTextRetriever(neo4j_uri, neo4j_user, neo4j_password)
        # Musimy użyć embeddera dla zapytania
        embedder = TextEmbedder(COHERE_API_KEY)  # Użyj globalnej zmiennej
        query_emb = embedder.get_text_embedding(query).tolist()
        
        results = hybrid.retrieve(query_emb, top_k=top_k)
        hybrid.close()
        
        # Pobierz teksty z wyników
        contexts = [result['data'].get('text', '') for result in results if 'text' in result['data']]
        
        # 3. Zbuduj prompt i wywołaj LLM
        limited_context = []
        total_chars = 0
        char_limit = 2000
        
        for ctx in contexts:
            if total_chars + len(ctx) <= char_limit:
                limited_context.append(ctx)
                total_chars += len(ctx)
            else:
                break
        
        context_text = "\n\n".join(limited_context)
        print(f"Użyto {len(context_text)} znaków kontekstu")
        
        # Przygotuj prompt dla Bielika
        documents_list = [{"snippet": chunk} for chunk in limited_context]
        documents_str = str(documents_list)
        
        system_prompt = (
            "Jesteś asystentem, który odpowiada na pytania w języku polskim. "
            f"Odpowiedz na pytanie używając tylko informacji z podanego kontekstu: {query} "
            f"{{documents={documents_str}}}"
        )
        
        # Generowanie odpowiedzi
        inputs = tokenizer(system_prompt, return_tensors="pt").to(model.device)
        
        print("\nGenerowanie odpowiedzi...")
        output = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            streamer=streamer
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return generated_text, contexts
    
    except Exception as e:
        print(f"Błąd podczas używania modelu Bielik: {e}")
        print("Upewnij się, że zainstalowano biblioteki: transformers, torch, bitsandbytes")
        return f"Błąd modelu: {str(e)}", []

def run_graph_maintenance(neo4j_uri, neo4j_user, neo4j_password, interval_hours=24):
    """
    Uruchamia okresową konserwację grafu w tle.
    """
    pruner = GraphPruner(neo4j_uri, neo4j_user, neo4j_password)
    
    def maintenance_job():
        while True:
            print("Uruchamianie konserwacji grafu...")
            try:
                pruner.run_maintenance()
                print("Konserwacja grafu zakończona.")
            except Exception as e:
                print(f"Błąd podczas konserwacji grafu: {e}")
            time.sleep(interval_hours * 3600)
    
    thread = threading.Thread(target=maintenance_job, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    # Parsowanie argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description="RAG System z Neo4j i wyborem modelu LLM")
    parser.add_argument('--model', type=str, choices=['cohere', 'bielik'], default='cohere',
                      help='Model LLM do użycia (cohere lub bielik)')
    parser.add_argument('--file', type=str, default=None,
                      help='Ścieżka do pliku tekstowego do przetworzenia')
    args = parser.parse_args()

    # Configuration
    COHERE_API_KEY = "xSywHzTHlEcq51tOI8rpxwWwtDdQnio5H7pPnuxs"
    NEO4J_URI = "neo4j+s://335a260d.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "4RMc8un8Yjzx9oy3_l2fDw5pNbuKuNdGjLHFI4a_EEU"
    
    # Path to data file
    if args.file:
        file_path = args.file
    else:
        file_path = os.path.join(os.path.dirname(__file__), "..", "data", "input1", "data1.txt")
    
    # Wyświetl informacje o wybranym modelu
    print(f"Używany model LLM: {args.model}")
    if args.model == 'bielik':
        print("Uwaga: Model Bielik wymaga dodatkowych bibliotek: transformers, torch, bitsandbytes")
    
    # Uruchom konserwację grafu w tle (co 24 godziny)
    maintenance_thread = run_graph_maintenance(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Process the file
    save_chunks_to_neo4j(file_path, COHERE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Interaktywny tryb zapytań
    while True:
        query = input("\nZadaj pytanie dotyczące zapisanych treści (lub 'q' aby wyjść): ")
        if query.lower() == 'q':
            break
        
        # Wykonaj RAG w zależności od wybranego modelu
        if args.model == 'cohere':
            response_text, sources = rag_query_cohere(query, COHERE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            print("\nOdpowiedź Cohere:\n", response_text)
        else:  # bielik
            response_text, sources = rag_query_bielik(query, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            print("\nOdpowiedź Bielik:\n", response_text)
            
        # Wyświetl źródła kontekstu
        print("\nŹródła kontekstu:")
        for idx, txt in enumerate(sources, 1):
            print(f"{idx}. {txt[:100]}...")