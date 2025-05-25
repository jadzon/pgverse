from neo4j import GraphDatabase

class Neo4jConnector:
    """
    Manages Neo4j driver connection.
    """
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_driver(self):
        return self.driver

    def close(self):
        self.driver.close()


class TextGraphBuilder:
    """
    Builds and maintains a self-learning graph of text nodes.
    Each node has:
      - id: unique identifier
      - type: data type, e.g. 'text', 'image', 'table'
      - text: textual content or metadata description
      - path: optional path to external data (e.g., image file, table CSV)
      - embedding: vector representation for similarity calculations
    Self-learning: reinforces edges when nodes are accessed together.
    """
    def __init__(self, connector: Neo4jConnector, similarity_threshold: float = 0.75):
        self.driver = connector.get_driver()
        self.threshold = similarity_threshold

    def close(self):
        self.driver.close()

    def insert_node(self, node_id: str, data_type: str, text: str, embedding: list = None, path: str = None):
        """
        Inserts or updates a Chunk node with given properties, including optional embedding.
        """
        query = '''
        MERGE (n:Chunk {id: $id})
        SET n.type = $type,
            n.text = $text,
            n.embedding = $embedding,
            n.path = $path
        '''
        with self.driver.session() as session:
            session.run(query, id=node_id, type=data_type, text=text, embedding=embedding, path=path)

    def create_text_relations(self):
        """
        Create or update SIMILAR_TO relationships between text nodes based
        on cosine similarity of embedding arrays stored in property 'embedding'.
        (Assumes embedding property exists on each node.)
        """
        query = '''
        MATCH (a:Chunk), (b:Chunk)
        WHERE a.type = 'text' AND b.type = 'text' AND elementId(a) < elementId(b)
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
        MERGE (a)-[r:SIMILAR_TO]->(b)
        ON CREATE SET r.weight = sim, r.last_used = timestamp()
        ON MATCH SET r.weight = sim, r.last_used = timestamp()
        '''
        with self.driver.session() as session:
            session.run(query, threshold=self.threshold)

    def prune_old(self, max_age_ms: int = 30*24*3600*1000):
        """
        Delete relationships not updated within max_age_ms.
        """
        query = '''
        MATCH ()-[r:SIMILAR_TO]->()
        WHERE r.last_used IS NOT NULL AND r.last_used < timestamp() - $max_age
        DELETE r
        '''
        with self.driver.session() as session:
            session.run(query, max_age=max_age_ms)

    def run_maintenance(self):
        """Run periodic maintenance: create relations and prune old."""
        self.create_text_relations()
        self.prune_old()


class HybridTextRetriever:
    """
    Retriever tekstowy używający podobieństwa kosinusowego z grafem Neo4j.
    Obsługuje zarówno węzły Chunk jak i TextChunk dla kompatybilności.
    """
    def __init__(self, connector: Neo4jConnector):
        self.driver = connector.get_driver()

    def close(self):
        """Zamyka połączenie z bazą danych."""
        self.driver.close()

    def search_by_text(self, query_embedding: list, top_k: int = 5, score_threshold: float = 0.0, use_relations: bool = True):
        """
        Wyszukuje najbardziej podobne chunki na podstawie embeddingu zapytania.
        
        Args:
            query_embedding: Embedding zapytania jako lista floatów
            top_k: Maksymalna liczba wyników
            score_threshold: Minimalny próg podobieństwa
            use_relations: Czy użyć relacji grafu do rozszerzenia wyników
            
        Returns:
            Lista słowników z wynikami
        """
        # Najpierw spróbuj nowy format (Chunk)
        chunk_query = '''
        MATCH (n:Chunk)
        WHERE n.type = 'text' AND n.embedding IS NOT NULL
        WITH n,
        reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) |
            dot + n.embedding[i] * $query_emb[i]
        ) /
        (
            sqrt(reduce(norm_n = 0.0, i IN range(0, size(n.embedding)-1) |
                norm_n + n.embedding[i] * n.embedding[i]
            )) *
            sqrt(reduce(norm_q = 0.0, i IN range(0, size($query_emb)-1) |
                norm_q + $query_emb[i] * $query_emb[i]
            ))
        ) AS score
        WHERE score > $threshold
        RETURN n.id as id, n.text as text, n.path as source, score, 'Chunk' as node_type
        ORDER BY score DESC
        LIMIT $top_k
        '''
        
        # Jeśli nie ma wyników, spróbuj stary format (TextChunk)
        textchunk_query = '''
        MATCH (n:TextChunk)
        WHERE n.embedding IS NOT NULL
        WITH n,
        reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) |
            dot + n.embedding[i] * $query_emb[i]
        ) /
        (
            sqrt(reduce(norm_n = 0.0, i IN range(0, size(n.embedding)-1) |
                norm_n + n.embedding[i] * n.embedding[i]
            )) *
            sqrt(reduce(norm_q = 0.0, i IN range(0, size($query_emb)-1) |
                norm_q + $query_emb[i] * $query_emb[i]
            ))
        ) AS score
        WHERE score > $threshold
        RETURN n.id as id, n.text as text, n.source as source, score, 'TextChunk' as node_type
        ORDER BY score DESC
        LIMIT $top_k
        '''
        
        results = []
        with self.driver.session() as session:
            # Spróbuj nowy format
            chunk_result = session.run(chunk_query, query_emb=query_embedding, threshold=score_threshold, top_k=top_k)
            chunk_records = list(chunk_result)
            
            if chunk_records:
                results = [{'data': {'id': record['id'], 'text': record['text'], 'source': record['source']}, 
                           'score': record['score'], 'node_type': record['node_type']} for record in chunk_records]
            else:
                # Jeśli brak wyników w nowym formacie, spróbuj stary
                textchunk_result = session.run(textchunk_query, query_emb=query_embedding, threshold=score_threshold, top_k=top_k)
                results = [{'data': {'id': record['id'], 'text': record['text'], 'source': record['source']}, 
                           'score': record['score'], 'node_type': record['node_type']} for record in textchunk_result]
        
        # Jeśli włączone są relacje, rozszerz wyniki
        if use_relations and results:
            results = self._expand_with_relations(results, top_k)
        
        return results

    def _expand_with_relations(self, initial_results: list, top_k: int):
        """
        Rozszerza wyniki o powiązane węzły poprzez relacje SIMILAR_TO.
        """
        expanded_results = initial_results.copy()
        processed_ids = {result['data']['id'] for result in initial_results}
        
        for result in initial_results:
            node_id = result['data']['id']
            node_type = result.get('node_type', 'TextChunk')
            
            # Znajdź powiązane węzły
            if node_type == 'Chunk':
                relation_query = '''
                MATCH (n:Chunk {id: $node_id})-[r:SIMILAR_TO]-(related:Chunk)
                WHERE related.type = 'text' AND related.id <> $node_id
                RETURN related.id as id, related.text as text, related.path as source, 
                       r.weight as relation_score, 'Chunk' as node_type
                ORDER BY r.weight DESC
                LIMIT 3
                '''
            else:
                relation_query = '''
                MATCH (n:TextChunk {id: $node_id})-[r:SIMILAR_TO]-(related:TextChunk)
                WHERE related.id <> $node_id
                RETURN related.id as id, related.text as text, related.source as source, 
                       r.weight as relation_score, 'TextChunk' as node_type
                ORDER BY r.weight DESC
                LIMIT 3
                '''
            
            with self.driver.session() as session:
                related_result = session.run(relation_query, node_id=node_id)
                
                for related_record in related_result:
                    related_id = related_record['id']
                    if related_id not in processed_ids and len(expanded_results) < top_k:
                        # Dodaj powiązany węzeł z nieco obniżonym score
                        expanded_results.append({
                            'data': {
                                'id': related_id,
                                'text': related_record['text'],
                                'source': related_record['source']
                            },
                            'score': related_record['relation_score'] * 0.8,  # Obniż score dla relacji
                            'node_type': related_record['node_type'],
                            'relation': 'SIMILAR_TO'
                        })
                        processed_ids.add(related_id)
        
        # Posortuj wszystkie wyniki według score
        expanded_results.sort(key=lambda x: x['score'], reverse=True)
        return expanded_results[:top_k]

    def retrieve(self, query_embedding: list, top_k: int = 5, score_threshold: float = 0.0):
        """
        Metoda kompatybilna z starą implementacją.
        """
        return self.search_by_text(query_embedding, top_k, score_threshold, use_relations=False)


class ImageRetriever:
    """
    Retriever dla obrazów używający podobieństwa kosinusowego z grafem Neo4j.
    """
    def __init__(self, connector: Neo4jConnector):
        self.driver = connector.get_driver()

    def close(self):
        """Zamyka połączenie z bazą danych."""
        self.driver.close()

    def search_by_image(self, query_embedding: list, top_k: int = 5, score_threshold: float = 0.66):
        """
        Wyszukuje najbardziej podobne obrazy na podstawie embeddingu zapytania.
        
        Args:
            query_embedding: Embedding zapytania jako lista floatów
            top_k: Maksymalna liczba wyników
            score_threshold: Minimalny próg podobieństwa
            
        Returns:
            Lista słowników z wynikami
        """
        query = '''
        MATCH (n:Chunk)
        WHERE n.type = 'image' AND n.embedding IS NOT NULL
        WITH n,
        reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) |
            dot + n.embedding[i] * $query_emb[i]
        ) /
        (
            sqrt(reduce(norm_n = 0.0, i IN range(0, size(n.embedding)-1) |
                norm_n + n.embedding[i] * n.embedding[i]
            )) *
            sqrt(reduce(norm_q = 0.0, i IN range(0, size($query_emb)-1) |
                norm_q + $query_emb[i] * $query_emb[i]
            ))
        ) AS score
        WHERE score > $threshold
        RETURN n.id as id, n.text as description, n.path as image_path, score
        ORDER BY score DESC
        LIMIT $top_k
        '''
        
        with self.driver.session() as session:
            result = session.run(query, query_emb=query_embedding, threshold=score_threshold, top_k=top_k)
            return [{'data': {'id': record['id'], 'description': record['description'], 'path': record['image_path']}, 
                    'score': record['score']} for record in result]


class GraphPruner:
    """
    Klasa do konserwacji grafu - usuwa stare relacje.
    """
    def __init__(self, connector: Neo4jConnector, similarity_threshold: float = 0.7):
        self.connector = connector
        self.text_graph = TextGraphBuilder(connector, similarity_threshold)
    
    def run_maintenance(self):
        """Uruchamia konserwację grafu."""
        self.text_graph.run_maintenance()
    
    def close(self):
        """Zamyka połączenie z bazą danych."""
        self.text_graph.close()