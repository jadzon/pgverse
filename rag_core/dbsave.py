import os
import sys
from neo4j import GraphDatabase
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag_functions.embedding_chunker import CohereTextChunker
from rag_functions.embeddings import TextEmbedder
from rag_functions.graph import GraphBuilder

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

if __name__ == "__main__":
    # Configuration
    # You would typically get these from environment variables or a config file
    COHERE_API_KEY = "xSywHzTHlEcq51tOI8rpxwWwtDdQnio5H7pPnuxs"  # Replace with your actual API key
    NEO4J_URI="neo4j+s://335a260d.databases.neo4j.io"
    NEO4J_USER="neo4j"
    NEO4J_PASSWORD="4RMc8un8Yjzx9oy3_l2fDw5pNbuKuNdGjLHFI4a_EEU"
    
    # Path to data file
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "input1", "data1.txt")
    
    # Process the file
    save_chunks_to_neo4j(file_path, COHERE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)