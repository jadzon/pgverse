import os
import sys
from neo4j import GraphDatabase
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cohere
import threading
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

from rag_functions.embedding_chunker import CohereTextChunker
from rag_functions.embeddings import TextEmbedder, ImageEmbedder, CLIPEmbedder
from rag_functions.graph import (
    Neo4jConnector, TextGraphBuilder, HybridTextRetriever, 
    ImageRetriever, GraphPruner
)
from rag_functions.cleaner import full_database_cleanup


def save_chunks_to_neo4j(file_path, api_key, neo4j_uri, neo4j_user, neo4j_password, max_tokens=500):
    """
    Chunks text or image, creates embeddings, and saves to Neo4j with type tagging
    
    Args:
        file_path: Path to the text or image file
        api_key: Cohere API key
        neo4j_uri: URI for Neo4j database
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        max_tokens: Maximum tokens per chunk (text only)
    """
    _, ext = os.path.splitext(file_path.lower())
    if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
        data_type = 'image'
    else:
        data_type = 'text'

    # Initialize chunker and appropriate embedder based on data type
    if data_type == 'text':
        chunker = CohereTextChunker(api_key)
        embedder = CLIPEmbedder()
    else:
        # For images, use ImageEmbedder
        embedder = CLIPEmbedder()

    # Connect to Neo4j using new connector
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    graph_builder = TextGraphBuilder(connector)

    # Read and chunk the data
    if data_type == 'text':
        print(f"Chunking text from {file_path}...")
        chunks = chunker.chunk_from_file(file_path, max_tokens)
    else:
        print(f"Processing image {file_path} as single chunk...")
        chunks = [file_path]
    print(f"Created {len(chunks)} chunk(s)")

    # Save embeddings to Neo4j
    print("Generating embeddings and saving to Neo4j...")
    saved_count = 0
    failed_count = 0
    
    try:
        with connector.get_driver().session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            )
            
        for i, chunk in enumerate(chunks):
            try:
                # Create a unique ID
                base = os.path.basename(file_path)
                chunk_id = f"chunk_{base}_{i}"
                
                # Zapisz dane w zależności od typu zawartości
                if data_type == 'text':
                    embedding = embedder.get_text_embedding(chunk).tolist()
                    # Use TextGraphBuilder to insert node
                    graph_builder.insert_node(
                        node_id=chunk_id,
                        data_type=data_type,
                        text=chunk,
                        embedding=embedding,
                        path=file_path
                    )
                else:
                    # Dla obrazów, zapisz ścieżkę do pliku
                    embedding = embedder.get_image_embedding(chunk).tolist()
                    graph_builder.insert_node(
                        node_id=chunk_id,
                        data_type=data_type,
                        text="",  # Empty text for images
                        embedding=embedding,
                        path=chunk
                    )
                    
                saved_count += 1
                
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    print(f"Saved {saved_count}/{len(chunks)} chunks")
                    
            except Exception as e:
                failed_count += 1
                print(f"Błąd podczas zapisywania chunka {i}: {str(e)}")
        
        print(f"Podsumowanie: zapisano {saved_count}/{len(chunks)} chunków, błędów: {failed_count}")
    finally:
        connector.close()

def rag_query_cohere(query, api_key, neo4j_uri, neo4j_user, neo4j_password, top_k=3):
    """
    Wykonuje retrieval z Neo4j na podstawie embeddingu zapytania,
    wyszukuje pasujące obrazy i generuje odpowiedź za pomocą Cohere Chat.
    """
    co = cohere.Client(api_key)
    # 1. Embed zapytanie używając CLIPEmbedder
    embedder = CLIPEmbedder()
    query_emb = embedder.get_text_embedding(query).tolist()

    # 2. Pobierz top_k najbardziej podobnych chunków tekstowych z Neo4j
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    text_retriever = HybridTextRetriever(connector)
    
    try:
        # Use HybridTextRetriever for text search
        text_results = text_retriever.search_by_text(query_emb, top_k=top_k, score_threshold=0.25)
        contexts = [result['data']['text'] for result in text_results if 'text' in result['data']]
        
        # 2a. Wyszukaj obrazy podobne do zapytania
        image_retriever = ImageRetriever(connector)
        image_results = image_retriever.search_by_image(query_emb, top_k=3, score_threshold=0.3)
        
        # Filtruj obrazy z wyższym progiem podobieństwa
        relevant_images = []
        for result in image_results:
            if result['score'] > 0.3 and 'path' in result['data']:
                relevant_images.append({
                    'path': result['data']['path'],
                    'score': result['score']
                })
            print(f"Znaleziony obraz: {result['data'].get('path', 'brak_ścieżki')} (podobieństwo: {result['score']:.4f})")

        # Dodaj dodatkowe wyszukiwanie przez relacje grafowe
        if len(relevant_images) < 2:
            try:
                with connector.get_driver().session() as session:
                    # Znajdź obrazy połączone z tekstem, który pasuje do zapytania
                    related_img_result = session.run("""
                    MATCH (t:Chunk {type: 'text'})-[:TEXT_ILLUSTRATED_BY]->(img:Chunk {type: 'image'})
                    WITH t, img, vector.similarity.cosine(t.embedding, $query_emb) AS score
                    WHERE score > 0.4
                    RETURN img.path AS path, score
                    ORDER BY score DESC
                    LIMIT 2
                    """, query_emb=query_emb)
                    
                    for record in related_img_result:
                        path = record["path"]
                        score = record["score"]
                        # Sprawdź czy obraz nie jest już dodany
                        if not any(img['path'] == path for img in relevant_images):
                            print(f"Dodano obraz przez relacje grafowe: {path} (podobieństwo: {score:.4f})")
                            relevant_images.append({
                                'path': path,
                                'score': score * 0.95
                            })
            except Exception as e:
                print(f"Błąd podczas wyszukiwania obrazów przez relacje: {e}")

        # 3. Zbuduj prompt i wywołaj LLM - z ponumerowanymi źródłami
        limited_context = []
        total_chars = 0
        char_limit = 2000
        
        for i, ctx in enumerate(contexts, 1):
            if total_chars + len(ctx) <= char_limit:
                limited_context.append(f"[Źródło {i}]: {ctx}")
                total_chars += len(ctx) + 15
            else:
                remaining = char_limit - total_chars
                if remaining > 100:
                    limited_context.append(f"[Źródło {i}]: {ctx[:remaining]} ...")
                break
        
        context_text = "\n\n".join(limited_context)
        print(f"Użyto {len(context_text)} znaków kontekstu")
        
        # Dodajemy informację o znalezionych obrazach do kontekstu
        images_info = ""
        if relevant_images:
            images_info = "\n\nZnalezione powiązane obrazy:\n"
            for i, img in enumerate(relevant_images, 1):
                images_info += f"[Obraz {i}]: {img['path']} (podobieństwo: {img['score']:.4f})\n"
        
        # Wywołaj model Cohere z ulepszonym promptem wymagającym cytowania
        chat_resp = co.chat(
            message=f"Question: {query}",
            model="command",
            preamble=f"You are a helpful assistant that responds based on the provided context. "
                f"If you cannot find an answer in the context, ACKNOWLEDGE IT and respond 'Based on the available context, I cannot answer this question.' "
                f"After each sentence or statement, YOU MUST provide the source in square brackets, "
                f"e.g., [Source 1], [Source 2], etc. "
                f"If the information comes from multiple sources, list all of them, e.g., [Source 1, Source 3]. "
                f"When referencing images, mention them as [Image 1], [Image 2], etc. "
                f"If you cannot find an answer in the context, acknowledge it instead of making up information.\n\n{context_text}{images_info}"
        )
        
        # Pobierz odpowiedź z modelu
        model_response = chat_resp.text
        
        # Zawsze dodawaj sekcję z obrazami na końcu odpowiedzi
        full_response = model_response
        
        if relevant_images:
            if "Referenced Images" not in full_response:
                full_response += "\n\n## Powiązane obrazy:\n"
                for i, img in enumerate(relevant_images, 1):
                    full_response += f"**Obraz {i}**: {img['path']} (podobieństwo: {img['score']:.4f})\n"
        else:
            full_response += "\n\nNie znaleziono pasujących obrazów."
        
        return full_response, contexts, relevant_images
    
    finally:
        text_retriever.close()
        connector.close()

def rag_query_bielik(query, neo4j_uri, neo4j_user, neo4j_password, model_name="speakleash/Bielik-11B-v2.3-Instruct", top_k=3):
    """
    Wykonuje retrieval z Neo4j, wyszukuje pasujące obrazy 
    i generuje odpowiedź za pomocą modelu Bielik.
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
        connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
        
        # Znajdź chunki za pomocą hybrydowego wyszukiwania
        hybrid = HybridTextRetriever(connector)
        # Musimy użyć embeddera dla zapytania
        embedder = TextEmbedder(COHERE_API_KEY)  # Użyj globalnej zmiennej
        query_emb = embedder.get_text_embedding(query).tolist()
        
        try:
            results = hybrid.retrieve(query_emb, top_k=top_k)
            
            # Pobierz teksty z wyników
            contexts = [result['data'].get('text', '') for result in results if 'text' in result['data']]
            
            # 2a. Wyszukaj obrazy podobne do zapytania
            image_retriever = ImageRetriever(connector)
            image_results = image_retriever.search_by_image(query_emb, top_k=5, score_threshold=0.85)
            
            # Filtruj obrazy o podobieństwie > 0.85
            relevant_images = []
            for result in image_results:
                if result['score'] > 0.85 and 'path' in result['data']:
                    relevant_images.append({
                        'path': result['data']['path'],
                        'score': result['score']
                    })
            
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
            
            # Dodaj informację o obrazach, jeśli znaleziono
            images_info = ""
            if relevant_images:
                images_info = "\nZnaleziono powiązane obrazy: "
                for i, img in enumerate(relevant_images, 1):
                    images_info += f"{i}. {img['path']} (podobieństwo: {img['score']:.4f}); "
            
            # Przygotuj prompt dla Bielika
            documents_list = [{"snippet": chunk} for chunk in limited_context]
            documents_str = str(documents_list)
            
            system_prompt = (
                "Jesteś asystentem, który odpowiada na pytania w języku polskim. "
                f"Odpowiedz na pytanie używając tylko informacji z podanego kontekstu: {query} "
                f"{{documents={documents_str}}}{' ' + images_info if relevant_images else ''}"
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
            
            # Dodaj informację o obrazach do odpowiedzi
            if relevant_images:
                image_details = "\n\nZnalezione powiązane obrazy:\n"
                for i, img in enumerate(relevant_images, 1):
                    image_details += f"{i}. {img['path']} (podobieństwo: {img['score']:.4f})\n"
                generated_text += "\n" + image_details
            
            return generated_text, contexts, relevant_images
        
        finally:
            hybrid.close()
            connector.close()
    
    except Exception as e:
        print(f"Błąd podczas używania modelu Bielik: {e}")
        print("Upewnij się, że zainstalowano biblioteki: transformers, torch, bitsandbytes")
        return f"Błąd modelu: {str(e)}", [], []

def run_graph_maintenance(neo4j_uri, neo4j_user, neo4j_password, interval_hours=24):
    """
    Uruchamia okresową konserwację grafu w tle.
    """
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    pruner = GraphPruner(connector)
    
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

def process_images_folder(base_folder, api_key, neo4j_uri, neo4j_user, neo4j_password):
    """
    Przetwarza wszystkie obrazy z podfolderu 'images' i dodaje je do bazy Neo4j
    
    Args:
        base_folder: Główny folder zawierający podfolder 'images'
        api_key: Klucz API Cohere
        neo4j_uri: URI bazy Neo4j
        neo4j_user: Nazwa użytkownika Neo4j
        neo4j_password: Hasło Neo4j
    """
    images_folder = os.path.normpath(os.path.join(base_folder, 'images'))
    
    # Sprawdź czy folder istnieje
    if not os.path.exists(images_folder):
        print(f"Folder {images_folder} nie istnieje. Pomijam przetwarzanie obrazów.")
        return
    
    print(f"Znaleziono folder ze zdjęciami: {images_folder}")
    
    # Lista wspieranych formatów plików obrazów
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    
    # Znajdź wszystkie pliki obrazów w katalogu
    image_files = []
    for file in os.listdir(images_folder):
        _, ext = os.path.splitext(file.lower())
        if ext in supported_formats:
            image_files.append(os.path.join(images_folder, file))
    
    if not image_files:
        print("Nie znaleziono żadnych plików obrazów do przetworzenia.")
        return
    
    print(f"Znaleziono {len(image_files)} plików obrazów do przetworzenia.")
    
    # Przetwórz każdy obraz
    for img_path in image_files:
        print(f"Przetwarzanie obrazu: {img_path}")
        try:
            save_chunks_to_neo4j(img_path, api_key, neo4j_uri, neo4j_user, neo4j_password)
        except Exception as e:
            print(f"Błąd podczas przetwarzania obrazu {img_path}: {e}")
    
    print("Zakończono przetwarzanie wszystkich obrazów.")

def build_knowledge_graph(neo4j_uri, neo4j_user, neo4j_password, similarity_threshold=0.85):
    """
    Buduje graf wiedzy tworząc relacje między węzłami na podstawie podobieństwa.
    
    Args:
        neo4j_uri: URI do bazy danych Neo4j
        neo4j_user: Nazwa użytkownika Neo4j
        neo4j_password: Hasło Neo4j
        similarity_threshold: Próg podobieństwa (zwiększony z 0.75 na 0.85)
    """
    print("\nBudowanie grafu wiedzy i relacji semantycznych...")
    print(f"Używany próg podobieństwa: {similarity_threshold}")
    
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    graph_builder = TextGraphBuilder(connector, similarity_threshold)
    
    try:
        # Tworzenie relacji między tekstami
        graph_builder.create_text_relations()
        print("Utworzono relacje między węzłami:")
        print(" - SIMILAR_TO: połączenia między podobnymi fragmentami tekstu")
        
        # Dodatkowo można dodać relacje między obrazami i tekstem
        with connector.get_driver().session() as session:
            # Twórz relacje IMAGE_SIMILAR między podobnymi obrazami
            session.run("""
            MATCH (a:Chunk), (b:Chunk)
            WHERE a.type = 'image' AND b.type = 'image' AND elementId(a) < elementId(b)
              AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
            WITH a, b,
            reduce(dot = 0.0, i IN range(0, size(a.embedding)-1) |
                dot + a.embedding[i] * b.embedding[i]
            ) /
            (
                sqrt(reduce(na = 0.0, i IN range(0, size(a.embedding)-1) |
                    na + a.embedding[i] * a.embedding[i]
                )) *
                sqrt(reduce(nb = 0.0, i IN range(0, size(b.embedding)-1) |
                    nb + b.embedding[i] * b.embedding[i]
                ))
            ) AS sim
            WHERE sim >= $threshold
            MERGE (a)-[r:IMAGE_SIMILAR]->(b)
            ON CREATE SET r.weight = sim, r.last_used = timestamp()
            ON MATCH SET r.weight = sim, r.last_used = timestamp()
            """, threshold=similarity_threshold)
            
            # Twórz relacje między obrazami a tekstem
            session.run("""
            MATCH (img:Chunk), (txt:Chunk)
            WHERE img.type = 'image' AND txt.type = 'text' 
              AND img.embedding IS NOT NULL AND txt.embedding IS NOT NULL
            WITH img, txt,
            reduce(dot = 0.0, i IN range(0, size(img.embedding)-1) |
                dot + img.embedding[i] * txt.embedding[i]
            ) /
            (
                sqrt(reduce(ni = 0.0, i IN range(0, size(img.embedding)-1) |
                    ni + img.embedding[i] * img.embedding[i]
                )) *
                sqrt(reduce(nt = 0.0, i IN range(0, size(txt.embedding)-1) |
                    nt + txt.embedding[i] * txt.embedding[i]
                ))
            ) AS sim
            WHERE sim >= $threshold
            MERGE (img)-[r:IMAGE_ILLUSTRATES]->(txt)
            ON CREATE SET r.weight = sim, r.last_used = timestamp()
            ON MATCH SET r.weight = sim, r.last_used = timestamp()
            MERGE (txt)-[r2:TEXT_ILLUSTRATED_BY]->(img)
            ON CREATE SET r2.weight = sim, r2.last_used = timestamp()
            ON MATCH SET r2.weight = sim, r.last_used = timestamp()
            """, threshold=similarity_threshold * 0.8)  # Niższy próg dla relacji obraz-tekst
        
        print(" - IMAGE_SIMILAR: połączenia między podobnymi obrazami")
        print(" - IMAGE_ILLUSTRATES: połączenia od obrazów do powiązanego tekstu")
        print(" - TEXT_ILLUSTRATED_BY: połączenia od tekstu do powiązanych obrazów")
        
    except Exception as e:
        print(f"Błąd podczas budowania grafu wiedzy: {e}")
    finally:
        graph_builder.close()
        connector.close()

if __name__ == "__main__":
    # Parsowanie argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description="RAG System z Neo4j i wyborem modelu LLM")
    parser.add_argument('--model', type=str, choices=['cohere', 'bielik'], default='cohere',
                      help='Model LLM do użycia (cohere lub bielik)')
    parser.add_argument('--file', type=str, default=None,
                      help='Ścieżka do pliku tekstowego lub obrazu do przetworzenia')
    args = parser.parse_args()

    # Configuration
    COHERE_API_KEY = "xSywHzTHlEcq51tOI8rpxwWwtDdQnio5H7pPnuxs"
    NEO4J_URI = "neo4j+s://335a260d.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "4RMc8un8Yjzx9oy3_l2fDw5pNbuKuNdGjLHFI4a_EEU"
    
    # CZYSZCZENIE BAZY DANYCH NA POCZĄTKU
    print("=" * 60)
    print("CZYSZCZENIE BAZY DANYCH")
    print("=" * 60)
    cleanup_stats = full_database_cleanup(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, silent=False)
    print(f"Pełne czyszczenie zakończone!")
    print("=" * 60)
    
    # Path to data file
    if args.file:
        file_path = os.path.normpath(args.file)
    else:
        file_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "input1", "data1.txt"))
    
    # Wyświetl informacje o wybranym modelu
    print(f"Używany model LLM: {args.model}")
    if args.model == 'bielik':
        print("Uwaga: Model Bielik wymaga dodatkowych bibliotek: transformers, torch, bitsandbytes")
    
    # Uruchom konserwację grafu w tle (co 24 godziny)
    maintenance_thread = run_graph_maintenance(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Process the file
    save_chunks_to_neo4j(file_path, COHERE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Przetwarzaj obrazy z podfolderu 'images'
    images_base_folder = os.path.normpath(os.path.dirname(file_path))
    process_images_folder(images_base_folder, COHERE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    build_knowledge_graph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, similarity_threshold=0.75)
    
    # Interaktywny tryb zapytań
    while True:
        query = input("\nZadaj pytanie dotyczące zapisanych treści (lub 'q' aby wyjść): ")
        if query.lower() == 'q':
            break
        
        # Wykonaj RAG w zależności od wybranego modelu
        if args.model == 'cohere':
            response_text, sources, images = rag_query_cohere(query, COHERE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            print("\nOdpowiedź Cohere:\n", response_text)
        else:  # bielik
            response_text, sources, images = rag_query_bielik(query, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            print("\nOdpowiedź Bielik:\n", response_text)
            
        # Wyświetl źródła kontekstu
        print("\nŹródła kontekstu tekstowego:")
        for idx, txt in enumerate(sources, 1):
            print(f"{idx}. {txt[:100]}...")
        
        # Jeśli znaleziono obrazy ale nie zostały już wyświetlone w odpowiedzi
        if images and "Znalezione powiązane obrazy:" not in response_text:
            print("\nZnalezione powiązane obrazy:")
            for idx, img in enumerate(images, 1):
                print(f"{idx}. {img['path']} (podobieństwo: {img['score']:.4f})")