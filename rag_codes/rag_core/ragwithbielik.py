import re
import cohere
from neo4j import GraphDatabase
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

nltk.download('punkt', quiet=True)
def load_polish_stopwords(filepath):
    return []


stopwords_filepath = 'polish_stopwords.txt'
polish_stop_words = load_polish_stopwords(stopwords_filepath)

def simple_sentence_tokenize(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def estimate_tokens(text):
    words = text.split()
    return len(words)

def split_long_segments(segments, max_tokens=500):
    result = []
    for segment in segments:
        if estimate_tokens(segment) <= max_tokens:
            result.append(segment)
            continue
            
        sentences = simple_sentence_tokenize(segment)
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            
            if sentence_tokens > max_tokens:
                if current_chunk:
                    result.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                    
                words = sentence.split()
                chunk = []
                chunk_tokens = 0
                
                for word in words:
                    if chunk_tokens + 1 > max_tokens:
                        result.append(" ".join(chunk))
                        chunk = [word]
                        chunk_tokens = 1
                    else:
                        chunk.append(word)
                        chunk_tokens += 1
                        
                if chunk:
                    result.append(" ".join(chunk))
            elif current_tokens + sentence_tokens > max_tokens:
                result.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                
        if current_chunk:
            result.append(" ".join(current_chunk))
            
    return result

def iterative_segmentation(text, max_tokens=500):
    sentences = simple_sentence_tokenize(text)
    if len(sentences) <= 2:
        if estimate_tokens(text) > max_tokens:
            return split_long_segments([text], max_tokens)
        return [text]
    
    vectorizer = TfidfVectorizer(stop_words=polish_stop_words, min_df=1, max_df=0.9)
    X = vectorizer.fit_transform(sentences)
    
    similarities = []
    for i in range(len(sentences) - 1):
        vec1 = X[i].toarray().flatten()
        vec2 = X[i+1].toarray().flatten()
        if np.sum(vec1) > 0 and np.sum(vec2) > 0:
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(sim)
        else:
            similarities.append(0)
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) > 1:
        print(f"Wykryto {len(paragraphs)} naturalnych paragrafów")
        paragraphs = split_long_segments(paragraphs, max_tokens)
        print(f"Po podziale długich paragrafów: {len(paragraphs)} segmentów")
        return paragraphs
    
    if similarities:
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        similarity_threshold = max(0.2, mean_sim - 0.5 * std_sim)
        print(f"Adaptacyjny próg podobieństwa: {similarity_threshold:.4f}")
    else:
        similarity_threshold = 0.3
    
    segments = []
    current_segment = [sentences[0]]
    current_topic_vector = X[0].toarray().flatten()
    current_tokens = estimate_tokens(sentences[0])
    
    for i in range(1, len(sentences)):
        sentence_vector = X[i].toarray().flatten()
        sentence_tokens = estimate_tokens(sentences[i])
        
        if current_tokens + sentence_tokens > max_tokens:
            segments.append(" ".join(current_segment))
            current_segment = [sentences[i]]
            current_topic_vector = sentence_vector
            current_tokens = sentence_tokens
            continue
        
        if np.sum(current_topic_vector) > 0 and np.sum(sentence_vector) > 0:
            similarity = np.dot(current_topic_vector, sentence_vector) / (
                np.linalg.norm(current_topic_vector) * np.linalg.norm(sentence_vector)
            )
        else:
            similarity = 0
        
        if similarity < similarity_threshold:
            if len(current_segment) >= 2 or len(segments) == 0:
                segments.append(" ".join(current_segment))
                current_segment = [sentences[i]]
                current_topic_vector = sentence_vector
                current_tokens = sentence_tokens
            else:
                current_segment.append(sentences[i])
                current_topic_vector = 0.4 * current_topic_vector + 0.6 * sentence_vector
                current_tokens += sentence_tokens
        else:
            current_segment.append(sentences[i])
            current_topic_vector = 0.7 * current_topic_vector + 0.3 * sentence_vector
            current_tokens += sentence_tokens
    
    if current_segment:
        segments.append(" ".join(current_segment))
    
    if len(segments) > len(sentences) // 3:
        print("Zbyt wiele segmentów, używam podejścia opartego na paragrafach")
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            paragraphs = split_long_segments(paragraphs, max_tokens)
            return paragraphs
    
    print(f"Wykryto {len(segments)} segmentów tematycznych (metoda iteracyjna)")
    segments = split_long_segments(segments, max_tokens)
    print(f"Po podziale długich segmentów: {len(segments)} segmentów")
    return segments

def parse_text_document(file_path, max_tokens=500):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    chapters = re.split(r'STAVE [IVX]+\.', text)
    chunks = []
    
    for chapter in chapters:
        if not chapter.strip():
            continue
            
        chapter_segments = iterative_segmentation(chapter, max_tokens)
        chunks.extend(chapter_segments)
    
    return chunks

def generate_embeddings(chunks, api_key):
    co = cohere.Client(api_key)
    embeddings = []
    for chunk in chunks:
        response = co.embed(texts=[chunk], model='embed-multilingual-v3.0', input_type="search_document")
        embeddings.append(response.embeddings[0])
    return embeddings

def store_in_neo4j(chunks, embeddings, uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        session.run("""
        CREATE VECTOR INDEX text_embeddings IF NOT EXISTS
        FOR (t:TextChunk) ON (t.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 1024,
                `vector.similarity_function`: 'cosine'
            }
        }
        """)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            session.run("""
            CREATE (t:TextChunk {
                id: $id,
                text: $text,
                position: $position,
                embedding: $embedding
            })
            """, id=f"chunk_{i}", text=chunk, position=i, embedding=embedding)
    driver.close()

def similarity_search(query_text, api_key, neo4j_uri, neo4j_username, neo4j_password, top_k=5):
    co = cohere.Client(api_key)
    query_embedding = co.embed(texts=[query_text], model='embed-multilingual-v3.0', input_type="search_query").embeddings[0]
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    with driver.session() as session:
        result = session.run("""
        MATCH (t:TextChunk)
        WITH t, vector.similarity.cosine(t.embedding, $query_embedding) AS score
        WHERE score > 0.7
        RETURN t.id AS id, t.text AS text, score
        ORDER BY score DESC LIMIT $top_k
        """, query_embedding=query_embedding, top_k=top_k)
        
        similar_chunks = [{"id": record["id"], "text": record["text"], "score": record["score"]}
                          for record in result]
    driver.close()
    return similar_chunks

def rerank_results(query, retrieved_chunks, api_key):
    co = cohere.Client(api_key)
    texts = [chunk["text"] for chunk in retrieved_chunks]
    reranked = co.rerank(
        query=query,
        documents=texts,
        top_n=len(texts),
        model="rerank-multilingual-v3.0"
    )
    
    reranked_chunks = []
    for result in reranked.results:
        original_chunk = retrieved_chunks[result.index]
        reranked_chunks.append({
            "id": original_chunk["id"],
            "text": original_chunk["text"],
            "score": result.relevance_score
        })
    return reranked_chunks

def calculate_metrics(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision * 100, recall * 100, f1 * 100

def process_query(query, file_path, cohere_api_key, neo4j_uri, neo4j_username, neo4j_password):
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    with driver.session() as session:
        count = session.run("MATCH (t:TextChunk) RETURN count(t) as count").single()["count"]
    driver.close()

    if count == 0:
        chunks = parse_text_document(file_path)
        embeddings = generate_embeddings(chunks, cohere_api_key)
        store_in_neo4j(chunks, embeddings, neo4j_uri, neo4j_username, neo4j_password)

    similar_chunks = similarity_search(query, cohere_api_key, neo4j_uri, neo4j_username, neo4j_password, top_k=10)

    reranked_chunks = rerank_results(query, similar_chunks, cohere_api_key)
    
    
    documents_list = [{"snippet": chunk["text"]} for chunk in reranked_chunks[:3]]
    documents_str = str(documents_list)

# Następnie użyj gotowego stringa w f-stringu
    system_prompt = (
        "Jesteś asystentem, który odpowiada na pytania dotyczące przedmiotu Podstawy automatyki w języku polskim. "
        f"Odpowiedz na pytanie używając tylko informacji z podanego kontekstu, w języku polskim: {query} "
        f"{{documents={documents_str}}}"
    )

    # Zmiana z 'prompt' na 'system_prompt'
    inputs = tokenizer(system_prompt, return_tensors="pt").to(model.device)

    # Generowanie odpowiedzi
    output = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        streamer=streamer
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)





    true_positives = sum(1 for chunk in reranked_chunks[:3] if chunk['score'] > 0.8)
    false_positives = sum(1 for chunk in reranked_chunks[:3] if chunk['score'] <= 0.8)
    false_negatives = 3 - true_positives

    precision, recall, f1 = calculate_metrics(true_positives, false_positives, false_negatives)

    return {
        "answer": generated_text,
        "sources": reranked_chunks[:3],
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    }

def plot_metrics(metrics_list):
    queries = range(1, len(metrics_list) + 1)
    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    f1_scores = [m['f1'] for m in metrics_list]

    plt.figure(figsize=(10, 6))
    plt.plot(queries, precisions, label='Precision', marker='o')
    plt.plot(queries, recalls, label='Recall', marker='s')
    plt.plot(queries, f1_scores, label='F1 Score', marker='^')
    plt.xlabel('Numer zapytania')
    plt.ylabel('Wartość metryki (%)')
    plt.title('Metryki wydajności systemu RAG')
    plt.legend()
    plt.grid(True)
    plt.savefig('rag_metrics.png')
    plt.close()

if __name__ == "__main__":
    file_path = "automatyka.txt"
    cohere_api_key = "xSywHzTHlEcq51tOI8rpxwWwtDdQnio5H7pPnuxs"
    neo4j_uri = "neo4j+s://34b5b39a.databases.neo4j.io"
    neo4j_username = "neo4j"
    neo4j_password = "gRnBKR2JKHoiWitvLViiloUZoNUHolGQbzBzl1aUKbo"

    

    results = []
    metrics_list = []

    #////////////////////////////////////////////////////////////////////////////////////////////////////
    #konfiguracja modelu
    #////////////////////////////////////////////////////////////////////////////////////////////////////

    model_name = "speakleash/Bielik-11B-v2.3-Instruct"  # Możesz zmienić na inny model
    input_file = "input.txt"
    output_file = "chunked_output.txt"  # Plik zbiorczy
    output_dir = "final_chunks"         # Katalog dla poszczególnych chunków
    load_4_bit = True

    quantization_config = None
    if load_4_bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print("Ładowanie modelu i tokenizera...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )
    
    print(f"Model {model_name} został załadowany.")
    print("Wpisz 'exit', 'quit' lub 'q' aby zakończyć program.")
    streamer = TextStreamer(tokenizer)

    #////////////////////////////////////////////////////////////////////////////////////////////////////
    #koniec konfiguracji modelu
    #////////////////////////////////////////////////////////////////////////////////////////////////////

    #////////////////////////////////////////////////////////////////////////////////////////////////////
    #pętla zapytań - start
    #////////////////////////////////////////////////////////////////////////////////////////////////////

    while True:
        # Pobierz prompt od użytkownika
        prompt = input("\nWprowadź prompt: ")
        
        # Sprawdź, czy użytkownik chce zakończyć program
        if prompt.lower() in ['exit', 'quit', 'q']:
            print("Kończenie programu...")
            break
        
        # Jeśli prompt jest pusty, poproś ponownie
        if not prompt.strip():
            print("Prompt nie może być pusty. Spróbuj ponownie.")
            continue
        
        # Przygotowanie i generowanie odpowiedzi
        print("\nGenerowanie odpowiedzi...")
        result = process_query(prompt, file_path, cohere_api_key, neo4j_uri, neo4j_username, neo4j_password)
        results.append(result)
        metrics_list.append(result['metrics'])

        # print(f"Odpowiedź: {result['answer']}")
        # print("Źródła:")
        # for source in result['sources']:
        #     print(f"- {source['text'][:100]}...")
        # print(f"Metryki: Precision: {result['metrics']['precision']:.2f}%, Recall: {result['metrics']['recall']:.2f}%, F1 Score: {result['metrics']['f1']:.2f}%")
        # print("\n")

    plot_metrics(metrics_list)
    print("Wykres metryk został zapisany jako 'rag_metrics.png'")

    #////////////////////////////////////////////////////////////////////////////////////////////////////
    #pętla zapytań - koniec
    #////////////////////////////////////////////////////////////////////////////////////////////////////
