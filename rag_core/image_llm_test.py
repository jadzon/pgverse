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
from rag_functions.embeddings import TextEmbedder, ImageEmbedder, CLIPEmbedder
from rag_functions.graph import GraphBuilder, HybridTextRetriever, GraphPruner, ImageRetriever, GraphCleaner


def graph_cleaner(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, operation="show"):
    """
    Zarządza relacjami w bazie danych Neo4j.
    
    Args:
        NEO4J_URI: URI do bazy danych Neo4j
        NEO4J_USER: Nazwa użytkownika Neo4j
        NEO4J_PASSWORD: Hasło Neo4j
        operation: Operacja do wykonania: "show", "delete_specific", "delete_all", "reset"
    """
    cleaner = GraphCleaner(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # Wyświetl aktualne liczby relacji w bazie danych
        relation_counts = cleaner.get_relationship_counts()
        print("Aktualne relacje w bazie danych:")
        for rel_type, count in relation_counts.items():
            print(f"  {rel_type}: {count}")

        if operation == "delete_specific":
            # Usuń tylko wybrane typy relacji
            deleted = cleaner.delete_specific_relationships(["SIMILAR_TO", "IMAGE_SIMILAR"])
            print(f"Usunięto {deleted} relacji.")
        elif operation == "delete_all":
            # Usuń wszystkie relacje
            deleted = cleaner.delete_all_relationships()
            print(f"Usunięto wszystkie relacje: {deleted}")
        elif operation == "reset":
            # Pełny reset bazy danych (tylko relacje, nie węzły)
            stats = cleaner.reset_database(keep_nodes=True)
            print(f"Reset bazy danych: usunięto {stats['relationships_deleted']} relacji i {stats['nodes_deleted']} węzłów.")
    finally:
        # Zawsze zamykaj połączenie
        cleaner.close()

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

    # Connect to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

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
    
    with driver.session() as session:
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
                    # Zapisz tekst z jawnie ustawioną właściwością "text"
                    session.run(
                        """
                        MERGE (c:Chunk {id: $id})
                        SET c.type = $type,
                            c.text = $value,
                            c.embedding = $embedding,
                            c.source = $source,
                            c.timestamp = timestamp()
                        """,
                        id=chunk_id,
                        type=data_type,
                        value=chunk,
                        embedding=embedding,
                        source=file_path
                    )
                else:
                    # Dla obrazów, zapisz ścieżkę do pliku z jawnie ustawioną właściwością "path"
                    embedding = embedder.get_image_embedding(chunk).tolist()
                    session.run(
                        """
                        MERGE (c:Chunk {id: $id})
                        SET c.type = $type,
                            c.path = $value,
                            c.embedding = $embedding,
                            c.source = $source,
                            c.timestamp = timestamp()
                        """,
                        id=chunk_id,
                        type=data_type,
                        value=chunk,
                        embedding=embedding,
                        source=file_path
                    )
                    
                saved_count += 1
                
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:  # Dodano warunek dla ostatniego chunka
                    print(f"Saved {saved_count}/{len(chunks)} chunks")
                    
            except Exception as e:
                failed_count += 1
                print(f"Błąd podczas zapisywania chunka {i}: {str(e)}")
        
        print(f"Podsumowanie: zapisano {saved_count}/{len(chunks)} chunków, błędów: {failed_count}")

def rag_query_cohere(query, api_key, neo4j_uri, neo4j_user, neo4j_password, top_k=3):
    """
    Wykonuje retrieval z Neo4j na podstawie embeddingu zapytania,
    wyszukuje pasujące obrazy i generuje odpowiedź za pomocą Cohere Chat.
    """
    co = cohere.Client(api_key)
    # 1. Embed zapytanie używając TextEmbedder, żeby mieć tę samą liczbę wymiarów
    embedder = CLIPEmbedder()
    query_emb = embedder.get_text_embedding(query).tolist()

    # 2. Pobierz top_k najbardziej podobnych chunków tekstowych z Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk)
            WHERE c.type = 'text'
            WITH c, vector.similarity.cosine(c.embedding, $query_emb) AS score
            WHERE score > 0.25
            RETURN c.text AS text, score
            ORDER BY score DESC
            LIMIT $top_k
        """, query_emb=query_emb, top_k=top_k)
        contexts = [record["text"] for record in result]
    driver.close()

    # 2a. Wyszukaj obrazy podobne do zapytania - zoptymalizowane dla lepszego wyszukiwania obrazów
    image_retriever = ImageRetriever(neo4j_uri, neo4j_user, neo4j_password)
    image_results = image_retriever.retrieve(query_emb, top_k=3, expand_k=2)
    image_retriever.close()
    
    # Filtruj obrazy z wyższym progiem podobieństwa
    relevant_images = []
    for result in image_results:
        # Zwiększony próg podobieństwa z 0.1 na 0.3
        if result['score'] > 0.3 and 'path' in result['data']:
            relevant_images.append({
                'path': result['data']['path'],
                'score': result['score']
            })
        print(f"Znaleziony obraz: {result['data'].get('path', 'brak_ścieżki')} (podobieństwo: {result['score']:.4f})")

    # Dodaj dodatkowe wyszukiwanie przez relacje grafowe
    if len(relevant_images) < 2:
        try:
            with driver.session() as session:
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
                            'score': score * 0.95  # Lekko obniż wynik dla obrazów znalezionych przez relacje
                        })
        except Exception as e:
            print(f"Błąd podczas wyszukiwania obrazów przez relacje: {e}")

    # Dodaj debugowanie
    print(f"Znaleziono {len(image_results)} potencjalnych obrazów")
    print(f"Z tego {len(relevant_images)} przekroczyło próg podobieństwa")
    
    # 3. Zbuduj prompt i wywołaj LLM - z ponumerowanymi źródłami
    limited_context = []
    total_chars = 0
    char_limit = 2000  # Przybliżony limit znaków (ok. 2000 tokenów)
    
    for i, ctx in enumerate(contexts, 1):
        if total_chars + len(ctx) <= char_limit:
            # Dodaj numer źródła przed każdym fragmentem kontekstu
            limited_context.append(f"[Źródło {i}]: {ctx}")
            total_chars += len(ctx) + 15  # +15 na oznaczenie źródła
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
    
    # Zawsze dodawaj sekcję z obrazami na końcu odpowiedzi, ale z lepszym formatowaniem
    full_response = model_response
    
    if relevant_images:
        if "Referenced Images" not in full_response:
            full_response += "\n\n## Powiązane obrazy:\n"
            for i, img in enumerate(relevant_images, 1):
                full_response += f"**Obraz {i}**: {img['path']} (podobieństwo: {img['score']:.4f})\n"
    else:
        full_response += "\n\nNie znaleziono pasujących obrazów."
    
    return full_response, contexts, relevant_images

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
        
        # 2a. Wyszukaj obrazy podobne do zapytania
        image_retriever = ImageRetriever(neo4j_uri, neo4j_user, neo4j_password)
        image_results = image_retriever.retrieve(query_emb, top_k=5)
        image_retriever.close()
        
        # Filtruj obrazy o podobieństwie > 0.24
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
    
    except Exception as e:
        print(f"Błąd podczas używania modelu Bielik: {e}")
        print("Upewnij się, że zainstalowano biblioteki: transformers, torch, bitsandbytes")
        return f"Błąd modelu: {str(e)}", [], []

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

# Dodaj nową funkcję do przetwarzania obrazów z podfolderu

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
    images_folder = os.path.join(base_folder, 'images')
    
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
    
    graph_builder = GraphBuilder(neo4j_uri, neo4j_user, neo4j_password, similarity_threshold)
    try:
        # Tworzenie relacji między obrazami oraz między obrazami i tekstem
        graph_builder.create_relationships()
        print("Utworzono relacje między węzłami:")
        print(" - IMAGE_SIMILAR: połączenia między podobnymi obrazami")
        print(" - IMAGE_ILLUSTRATES: połączenia od obrazów do powiązanego tekstu")
        print(" - TEXT_ILLUSTRATED_BY: połączenia od tekstu do powiązanych obrazów")
        print(" - SIMILAR_TO: ogólne relacje podobieństwa (dla zgodności wstecznej)")
    except Exception as e:
        print(f"Błąd podczas budowania grafu wiedzy: {e}")
    finally:
        graph_builder.close()

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
    
    graph_cleaner(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Process the file
    save_chunks_to_neo4j(file_path, COHERE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # Przetwarzaj obrazy z podfolderu 'images'
    process_images_folder(os.path.dirname(file_path), COHERE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    
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