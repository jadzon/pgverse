import os
import numpy as np
from neo4j import GraphDatabase

# --- TRUST LEVELS FOR EMBEDDING SOURCES (EDIT FREELY) ---
SOURCE_TRUST_LEVELS = {
    "wikipedia": 0.85,
    "book": 0.9,
    "scientific_article": 0.95,
    "blog": 0.5,
    "forum": 0.3,
    "social_media": 0.2,
    "news": 0.6,
    "unknown": 0.4  # Default trust for unknown sources
}

# --- CHOOSE DATA SOURCE TYPE FOR THIS RUN ---
source_type = "wikipedia"  # <--- Set here, e.g. "book", "blog", etc.

# --- SET DOC_ID HERE ---
doc_id = "my_document_2024"

# --- NEO4J CONNECTION CONFIG ---
neo4j_uri = "neo4j+s://335a260d.databases.neo4j.io"
neo4j_username = "neo4j"
neo4j_password = "4RMc8un8Yjzx9oy3_l2fDw5pNbuKuNdGjLHFI4a_EEU"

embedding_folder = "embeddings"

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

def save_embeddings(tx, doc_id, chunk_names, embeddings, weight):
    tx.run("MERGE (d:Document {id: $doc_id})", doc_id=doc_id)
    for chunk_name, embedding in zip(chunk_names, embeddings):
        tx.run("""
            MERGE (c:Chunk {doc_id: $doc_id, chunk_name: $chunk_name})
            SET c.embedding = $embedding
            WITH c
            MATCH (d:Document {id: $doc_id})
            MERGE (c)-[r:PART_OF_DOCUMENT]->(d)
            SET r.weight = $weight
        """, doc_id=doc_id, chunk_name=chunk_name, embedding=embedding.tolist(), weight=weight)

def main():
    chunk_files = [f for f in os.listdir(embedding_folder) if f.endswith('.npy')]
    chunk_files.sort()
    embeddings = []
    for file in chunk_files:
        path = os.path.join(embedding_folder, file)
        emb = np.load(path)
        embeddings.append(emb)

    # Get trust level for selected source_type
    weight = SOURCE_TRUST_LEVELS.get(source_type, SOURCE_TRUST_LEVELS["unknown"])

    with driver.session() as session:
        session.write_transaction(save_embeddings, doc_id, chunk_files, embeddings)

    print(f"Loaded and saved embeddings for document {doc_id} ({len(chunk_files)} chunks) with source '{source_type}' (trust level: {weight}).")

if __name__ == '__main__':
    main()
