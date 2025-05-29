import re
import cohere
from neo4j import GraphDatabase
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import json
from metrics.BERTscore import compute_bertscore

nltk.download('punkt', quiet=True)
def load_polish_stopwords(filepath):
    return []


stopwords_filepath = 'polish_stopwords.txt'
polish_stop_words = load_polish_stopwords(stopwords_filepath)



def generate_embeddings(chunks, api_key):
    co = cohere.Client(api_key)
    embeddings = []
    for chunk in chunks:
        response = co.embed(texts=[chunk], model='embed-multilingual-v3.0', input_type="search_document")
        embeddings.append(response.embeddings[0])
    return embeddings


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


def process_query(query, cohere_api_key, neo4j_uri, neo4j_username, neo4j_password,
                  output_bertscore_path='bertscore_results.json', output_chunks_path='chunks.json'):
    # 1. Pobierz podobne chunki
    retrieved = similarity_search(query, cohere_api_key, neo4j_uri, neo4j_username, neo4j_password, top_k=10)
    reranked = rerank_results(query, retrieved, cohere_api_key)

    # 2. Wydziel listę tekstów chuncków
    chunks = [chunk['text'] for chunk in reranked[:3]]

    # 2a. Zapisz chunki do pliku JSON
    with open(output_chunks_path, 'w', encoding='utf-8') as cf:
        json.dump({'chunks': chunks}, cf, ensure_ascii=False, indent=2)
    print(f"Zapisano chunki do {output_chunks_path}")

    # 3. Oblicz metryki BERTScore
    bert_results = compute_bertscore(query, chunks, lang='pl')

    # 4. Zapisz wyniki BERTScore do JSON
    ber_data = {'query': query, 'results': bert_results}
    with open(output_bertscore_path, 'w', encoding='utf-8') as bf:
        json.dump(ber_data, bf, ensure_ascii=False, indent=2)
    print(f"Zapisano BERTScore do {output_bertscore_path}")

    # 5. Generowanie odpowiedzi RAG
    inputs = tokenizer(query, return_tensors='pt').to(model.device)
        # 5. Generowanie odpowiedzi bez streamera
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # obetnij początek (czyli powtórzony prompt)
    response = full_text[len(query):].strip()
    return response





  

if __name__ == "__main__":
    file_path = "automatyka.txt"
    cohere_api_key = "qNDuEWEYe9GECJNq3v0FyNL3DNfKK1l8ddO3Ov5F"
    neo4j_uri = "neo4j+s://e12a1d85.databases.neo4j.io"
    neo4j_username = "neo4j"
    neo4j_password = "dbq34KlPGzGC5j4wC4ARZMJNIo5xBPWfW9koDBe99j0"

    

    results = []

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
        result = process_query(prompt,
                       cohere_api_key,
                       neo4j_uri,
                       neo4j_username,
                       neo4j_password)


        results.append(result)
        


    

    #////////////////////////////////////////////////////////////////////////////////////////////////////
    #pętla zapytań - koniec
    #////////////////////////////////////////////////////////////////////////////////////////////////////