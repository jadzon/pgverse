import cohere
from neo4j import GraphDatabase
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from rag_metrics.BERTscore import compute_and_save_bertscore


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
        result = session.run(
            """
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


def process_query(query,
                  cohere_api_key,
                  neo4j_uri,
                  neo4j_username,
                  neo4j_password,
                  tokenizer,
                  model):
    # 1. Retrieve and rerank chunks (jak było)
    retrieved = similarity_search(query, cohere_api_key, neo4j_uri, neo4j_username, neo4j_password, top_k=10)
    reranked = rerank_results(query, retrieved, cohere_api_key)
    top_chunks = reranked[:3]

    # 2. Przygotuj kontekst dla modelu (z rag_with_bertscore.py)
    documents_list = [{"snippet": chunk["text"]} for chunk in top_chunks]
    # system prompt w języku polskim, z przekazaniem kontekstu
    system_prompt = (
        "Jesteś asystentem, który odpowiada na pytania dotyczące przedmiotu Podstawy automatyki w języku polskim. "
        f"Odpowiedz na pytanie używając tylko informacji z podanego kontekstu.\n\n"
        f"Pytanie: {query}\n\n"
        f"Kontekst:\n{documents_list}\n\n"
    )

    streamer = TextStreamer(tokenizer)
    inputs = tokenizer(system_prompt, return_tensors='pt').to(model.device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    generated_ids = output_ids[0, prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if response.lower().startswith("odpowiedź:"):
        response = response[len("Odpowiedź:"):].strip()
    

    references = [c['text'] for c in top_chunks]
    compute_and_save_bertscore(query, response, references)
    # 5. Zwróć odpowiedź
    #print(f"Odpowiedź: {response}\n")
    return response


