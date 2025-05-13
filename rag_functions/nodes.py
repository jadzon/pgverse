import os
import numpy as np
from neo4j import GraphDatabase

# --- USTAW DOC_ID TUTAJ ---
doc_id = "moj_dokument_2024"  # <-- tu łatwo zmieniasz doc_id

# --- KONFIGURACJA POŁĄCZENIA ---
neo4j_uri = "neo4j+s://34b5b39a.databases.neo4j.io"
neo4j_username = "neo4j"
neo4j_password = "gRnBKR2JKHoiWitvLViiloUZoNUHolGQbzBzl1aUKbo"

embedding_folder = "embeddings"

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

def save_embeddings(tx, doc_id, chunk_names, embeddings):
    # Tworzymy/aktualizujemy węzeł dokumentu
    tx.run("MERGE (d:Document {id: $doc_id})", doc_id=doc_id)
    # Zapisujemy/aktualizujemy węzły chunków i relacje PART_OF_DOCUMENT
    for chunk_name, embedding in zip(chunk_names, embeddings):
        tx.run("""
            MERGE (c:Chunk {doc_id: $doc_id, chunk_name: $chunk_name})
            SET c.embedding = $embedding
            WITH c
            MATCH (d:Document {id: $doc_id})
            MERGE (c)-[:PART_OF_DOCUMENT]->(d)
        """, doc_id=doc_id, chunk_name=chunk_name, embedding=embedding.tolist())

def main():
    # Wczytaj embeddingi z folderu
    chunk_files = [f for f in os.listdir(embedding_folder) if f.endswith('.npy')]
    chunk_files.sort()
    embeddings = []
    for file in chunk_files:
        path = os.path.join(embedding_folder, file)
        emb = np.load(path)
        embeddings.append(emb)

    with driver.session() as session:
        session.execute_write(save_embeddings, doc_id, chunk_files, embeddings)

    print(f"Wczytano i zapisano embeddingi dla dokumentu {doc_id} ({len(chunk_files)} chunków).")

if __name__ == '__main__':
    main()
