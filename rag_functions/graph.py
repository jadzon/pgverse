from neo4j import GraphDatabase
import time
from collections import defaultdict


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
    def __init__(self, connector: Neo4jConnector, similarity_threshold: float = 0.8):
        self.driver = connector.get_driver()
        self.threshold = similarity_threshold
        self.usage_patterns = defaultdict(int)
        self.last_threshold_adjustment = time.time()

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
            n.path = $path,
            n.usage_count = COALESCE(n.usage_count, 0),
            n.created_at = COALESCE(n.created_at, timestamp()),
            n.last_accessed = timestamp()
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
        ON CREATE SET r.weight = sim, r.last_used = timestamp(), r.reinforcement_count = 0, r.created_at = timestamp()
        ON MATCH SET r.weight = sim, r.last_used = timestamp()
        '''
        with self.driver.session() as session:
            session.run(query, threshold=self.threshold)

    def reinforce_relationship(self, node_a_id: str, node_b_id: str, strength: float = 1.0):
        """
        Wzmacnia relację między dwoma węzłami na podstawie ich współwystępowania.
        Jeśli relacja nie istnieje, tworzy ją z małą wagą.
        """
        query = '''
        MATCH (a:Chunk {id: $node_a}), (b:Chunk {id: $node_b})
        MERGE (a)-[r:SIMILAR_TO]-(b)
        ON CREATE SET r.weight = 0.1, 
                     r.reinforcement_count = 1, 
                     r.last_used = timestamp(),
                     r.created_at = timestamp(),
                     r.learning_weight = 0.1
        ON MATCH SET r.weight = CASE 
                       WHEN r.weight + ($strength * 0.1) > 1.0 THEN 1.0
                       ELSE r.weight + ($strength * 0.1)
                     END,
                     r.reinforcement_count = COALESCE(r.reinforcement_count, 0) + 1,
                     r.last_used = timestamp(),
                     r.learning_weight = COALESCE(r.learning_weight, 0) + ($strength * 0.05)
        '''
        with self.driver.session() as session:
            session.run(query, node_a=node_a_id, node_b=node_b_id, strength=strength)

    def track_usage_pattern(self, node_ids: list, query_context: str = ""):
        """
        Śledzi wzorce użycia węzłów - które węzły są często używane razem.
        """
        if len(node_ids) < 2:
            return
            
        # Zwiększ licznik użycia dla każdego węzła
        for node_id in node_ids:
            self.increment_node_usage(node_id, query_context)
        
        # Wzmocnij relacje między wszystkimi parami węzłów
        for i, node_a in enumerate(node_ids):
            for node_b in node_ids[i+1:]:
                # Zapisz wzorzec lokalnie
                pattern_key = tuple(sorted([node_a, node_b]))
                self.usage_patterns[pattern_key] += 1
                
                # Wzmocnij relację w grafie
                strength = min(2.0, 1.0 + (self.usage_patterns[pattern_key] * 0.1))
                self.reinforce_relationship(node_a, node_b, strength)

    def increment_node_usage(self, node_id: str, context: str = ""):
        """
        Zwiększa licznik użycia węzła i zapisuje kontekst.
        """
        query = '''
        MATCH (n:Chunk {id: $node_id})
        SET n.usage_count = COALESCE(n.usage_count, 0) + 1,
            n.last_accessed = timestamp(),
            n.total_access_time = COALESCE(n.total_access_time, 0) + 1,
            n.contexts = CASE 
                WHEN n.contexts IS NULL THEN [$context]
                WHEN size(n.contexts) < 10 AND NOT $context IN n.contexts THEN n.contexts + $context
                ELSE n.contexts
            END
        '''
        with self.driver.session() as session:
            session.run(query, node_id=node_id, context=context[:100])  # Ogranicz długość kontekstu

    def decay_relationships(self, decay_factor: float = 0.95, min_weight: float = 0.1):
        """
        Zmniejsza wagę relacji, które nie były używane przez długi czas.
        Usuwa relacje poniżej minimalnej wagi.
        """
        thirty_days_ms = 30 * 24 * 60 * 60 * 1000
        seven_days_ms = 7 * 24 * 60 * 60 * 1000
        
        # Silny decay dla bardzo starych relacji
        strong_decay_query = '''
        MATCH ()-[r:SIMILAR_TO]->()
        WHERE r.last_used < timestamp() - $long_threshold
        SET r.weight = r.weight * $strong_decay,
            r.decay_applied = COALESCE(r.decay_applied, 0) + 1
        '''
        
        # Łagodny decay dla średnio starych relacji
        mild_decay_query = '''
        MATCH ()-[r:SIMILAR_TO]->()
        WHERE r.last_used < timestamp() - $short_threshold 
          AND r.last_used >= timestamp() - $long_threshold
        SET r.weight = r.weight * $mild_decay,
            r.decay_applied = COALESCE(r.decay_applied, 0) + 1
        '''
        
        # Usuń bardzo słabe relacje
        cleanup_query = '''
        MATCH ()-[r:SIMILAR_TO]->()
        WHERE r.weight < $min_weight AND r.reinforcement_count < 2
        DELETE r
        '''
        
        with self.driver.session() as session:
            # Zastosuj silny decay
            result1 = session.run(strong_decay_query, 
                                long_threshold=thirty_days_ms, 
                                strong_decay=decay_factor * 0.8)
            
            # Zastosuj łagodny decay  
            result2 = session.run(mild_decay_query,
                                short_threshold=seven_days_ms,
                                long_threshold=thirty_days_ms,
                                mild_decay=decay_factor)
            
            # Wyczyść słabe relacje
            result3 = session.run(cleanup_query, min_weight=min_weight)
            
            print(f"Zastosowano decay relacji. Usunięto {result3.consume().counters.relationships_deleted} słabych relacji.")

    def adaptive_threshold_adjustment(self):
        """
        Automatycznie dostosowuje próg podobieństwa na podstawie gęstości grafu.
        """
        # Sprawdź czy minęło wystarczająco czasu od ostatniej korekty
        current_time = time.time()
        if current_time - self.last_threshold_adjustment < 3600:  # 1 godzina
            return
            
        # Policz obecną liczbę relacji i węzłów
        stats_query = '''
        MATCH (n:Chunk) 
        WHERE n.type = 'text'
        OPTIONAL MATCH (n)-[r:SIMILAR_TO]-()
        RETURN count(DISTINCT n) as node_count, count(r) as relation_count,
               avg(r.weight) as avg_weight, max(r.weight) as max_weight
        '''
        
        with self.driver.session() as session:
            result = session.run(stats_query).single()
            node_count = result['node_count']
            relation_count = result['relation_count']
            avg_weight = result['avg_weight'] or 0
            max_weight = result['max_weight'] or 0
            
            if node_count > 1:
                max_possible_relations = node_count * (node_count - 1) / 2
                density = relation_count / max_possible_relations if max_possible_relations > 0 else 0
                
                old_threshold = self.threshold
                
                # Dostosuj próg na podstawie gęstości i jakości relacji
                if density > 0.3:  # Za dużo relacji
                    adjustment = min(0.05, (density - 0.3) * 0.1)
                    self.threshold = min(0.95, self.threshold + adjustment)
                elif density < 0.05:  # Za mało relacji
                    adjustment = min(0.05, (0.05 - density) * 0.2)
                    self.threshold = max(0.3, self.threshold - adjustment)
                
                # Dodatkowa korekta na podstawie jakości relacji
                if avg_weight > 0:
                    if avg_weight < 0.4:  # Słabe relacje
                        self.threshold = max(0.3, self.threshold - 0.02)
                    elif avg_weight > 0.8:  # Bardzo silne relacje
                        self.threshold = min(0.95, self.threshold + 0.02)
                
                self.last_threshold_adjustment = current_time
                
                if abs(old_threshold - self.threshold) > 0.01:
                    print(f"Próg podobieństwa dostosowany: {old_threshold:.3f} → {self.threshold:.3f}")
                    print(f"Gęstość grafu: {density:.3f}, Średnia waga: {avg_weight:.3f}")

    def create_semantic_clusters(self, min_cluster_size: int = 3, weight_threshold: float = 0.75):
        """
        Tworzy klastry semantyczne na podstawie silnych relacji.
        """
        query = '''
        MATCH (n:Chunk)-[r:SIMILAR_TO]-(m:Chunk)
        WHERE r.weight > $weight_threshold AND n.type = 'text'
        WITH n, collect(DISTINCT m) as connected_nodes
        WHERE size(connected_nodes) >= $min_size
        SET n.cluster_candidate = true,
            n.cluster_size = size(connected_nodes),
            n.cluster_strength = $weight_threshold
        RETURN n.id as node_id, size(connected_nodes) as cluster_size
        '''
        
        with self.driver.session() as session:
            result = session.run(query, weight_threshold=weight_threshold, min_size=min_cluster_size)
            clusters = [dict(record) for record in result]
            
            if clusters:
                print(f"Zidentyfikowano {len(clusters)} kandydatów na klastry semantyczne")
            
            return clusters

    def analyze_learning_patterns(self):
        """
        Analizuje wzorce uczenia się i zwraca statystyki.
        """
        # Statystyki węzłów
        node_stats_query = '''
        MATCH (n:Chunk)
        WHERE n.type = 'text'
        RETURN count(n) as total_nodes,
               avg(n.usage_count) as avg_usage,
               max(n.usage_count) as max_usage,
               count(CASE WHEN n.usage_count > 5 THEN 1 END) as popular_nodes
        '''
        
        # Statystyki relacji
        relation_stats_query = '''
        MATCH ()-[r:SIMILAR_TO]->()
        RETURN count(r) as total_relations,
               avg(r.weight) as avg_weight,
               avg(r.reinforcement_count) as avg_reinforcements,
               count(CASE WHEN r.weight > 0.8 THEN 1 END) as strong_relations,
               count(CASE WHEN r.reinforcement_count > 3 THEN 1 END) as learned_relations
        '''
        
        # Najczęściej używane węzły
        popular_nodes_query = '''
        MATCH (n:Chunk)
        WHERE n.type = 'text' AND n.usage_count > 0
        RETURN n.id as id, n.usage_count as usage_count, size(n.contexts) as context_variety
        ORDER BY n.usage_count DESC
        LIMIT 10
        '''
        
        # Najsilniejsze relacje
        strong_relations_query = '''
        MATCH (a:Chunk)-[r:SIMILAR_TO]-(b:Chunk)
        WHERE r.weight > 0.7
        RETURN a.id as node_a, b.id as node_b, r.weight as weight, 
               r.reinforcement_count as reinforcements
        ORDER BY r.weight DESC, r.reinforcement_count DESC
        LIMIT 10
        '''
        
        with self.driver.session() as session:
            node_stats = session.run(node_stats_query).single()
            relation_stats = session.run(relation_stats_query).single()
            popular_nodes = [dict(record) for record in session.run(popular_nodes_query)]
            strong_relations = [dict(record) for record in session.run(strong_relations_query)]
            
            return {
                'node_statistics': dict(node_stats),
                'relation_statistics': dict(relation_stats),
                'popular_nodes': popular_nodes,
                'strong_relations': strong_relations,
                'usage_patterns_count': len(self.usage_patterns),
                'current_threshold': self.threshold
            }

    def prune_old(self, max_age_ms: int = 30*24*3600*1000):
        """
        Delete relationships not updated within max_age_ms.
        Enhanced with learning-aware pruning.
        """
        # Nie usuwaj relacji z wysoką liczbą wzmocnień, nawet jeśli są stare
        query = '''
        MATCH ()-[r:SIMILAR_TO]->()
        WHERE r.last_used IS NOT NULL 
          AND r.last_used < timestamp() - $max_age
          AND (r.reinforcement_count IS NULL OR r.reinforcement_count < 3)
          AND (r.learning_weight IS NULL OR r.learning_weight < 0.3)
        DELETE r
        '''
        with self.driver.session() as session:
            result = session.run(query, max_age=max_age_ms)
            deleted_count = result.consume().counters.relationships_deleted
            if deleted_count > 0:
                print(f"Usunięto {deleted_count} starych relacji podczas czyszczenia")

    def run_maintenance(self):
        """
        Run periodic maintenance: create relations, apply decay, adjust thresholds, and prune old.
        """
        print("Rozpoczynam konserwację grafu z uczeniem się...")
        
        # Standardowe relacje
        self.create_text_relations()
        
        # Zastosuj zanikanie relacji
        self.decay_relationships()
        
        # Dostosuj próg adaptacyjnie
        self.adaptive_threshold_adjustment()
        
        # Znajdź klastry semantyczne
        self.create_semantic_clusters()
        
        # Wyczyść stare relacje
        self.prune_old()
        
        print("Konserwacja grafu zakończona.")


class HybridTextRetriever:
    """
    Retriever tekstowy używający podobieństwa kosinusowego z grafem Neo4j.
    Obsługuje zarówno węzły Chunk jak i TextChunk dla kompatybilności.
    Enhanced with learning capabilities.
    """
    def __init__(self, connector: Neo4jConnector):
        self.driver = connector.get_driver()
        self.graph_builder = None  # Will be set externally for learning

    def close(self):
        """Zamyka połączenie z bazą danych."""
        self.driver.close()

    def search_by_text(self, query_embedding: list, top_k: int = 5, score_threshold: float = 0.0, use_relations: bool = True):
        """
        Wyszukuje najbardziej podobne chunki na podstawie embeddingu zapytania.
        Enhanced with learning tracking.
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

    def search_by_text_with_learning(self, query_embedding: list, query_text: str = "", top_k: int = 5, score_threshold: float = 0.0):
        """
        Wyszukuje podobne chunki i uczy się z wzorców wyszukiwania.
        """
        results = self.search_by_text(query_embedding, top_k, score_threshold, use_relations=True)
        
        # Jeśli znaleziono wyniki i mamy dostęp do graph_builder, naucz się z tego wzorca
        if results and self.graph_builder:
            # Pobierz ID wszystkich zwróconych węzłów
            result_ids = [result['data']['id'] for result in results]
            
            # Przekaż wzorzec do TextGraphBuilder dla wzmocnienia relacji
            self.graph_builder.track_usage_pattern(result_ids, query_text[:50])
        
        return results

    def get_recommendation_based_on_history(self, recently_accessed_ids: list, top_k: int = 5):
        """
        Rekomenduje podobne węzły na podstawie historii dostępu.
        """
        if not recently_accessed_ids:
            return []
            
        query = '''
        MATCH (accessed:Chunk)-[r:SIMILAR_TO]-(recommended:Chunk)
        WHERE accessed.id IN $accessed_ids 
          AND NOT recommended.id IN $accessed_ids
          AND r.weight > 0.3
          AND recommended.type = 'text'
        RETURN recommended.id as id, recommended.text as text, recommended.path as source,
               avg(r.weight) as avg_similarity, count(r) as connection_count,
               max(r.reinforcement_count) as max_reinforcements
        ORDER BY avg_similarity DESC, connection_count DESC, max_reinforcements DESC
        LIMIT $top_k
        '''
        
        with self.driver.session() as session:
            result = session.run(query, accessed_ids=recently_accessed_ids, top_k=top_k)
            return [{'data': {'id': record['id'], 'text': record['text'], 'source': record['source']}, 
                    'score': record['avg_similarity'], 
                    'connections': record['connection_count'],
                    'reinforcements': record['max_reinforcements']} 
                    for record in result]

    def _expand_with_relations(self, initial_results: list, top_k: int):
        """
        Rozszerza wyniki o powiązane węzły poprzez relacje SIMILAR_TO.
        Enhanced to prefer learned relationships.
        """
        expanded_results = initial_results.copy()
        processed_ids = {result['data']['id'] for result in initial_results}
        
        for result in initial_results:
            node_id = result['data']['id']
            node_type = result.get('node_type', 'TextChunk')
            
            # Znajdź powiązane węzły, preferując te z wysokimi wzmocnieniami
            if node_type == 'Chunk':
                relation_query = '''
                MATCH (n:Chunk {id: $node_id})-[r:SIMILAR_TO]-(related:Chunk)
                WHERE related.type = 'text' AND related.id <> $node_id
                RETURN related.id as id, related.text as text, related.path as source, 
                       r.weight as relation_score, 'Chunk' as node_type,
                       COALESCE(r.reinforcement_count, 0) as reinforcements
                ORDER BY r.weight DESC, reinforcements DESC
                LIMIT 3
                '''
            else:
                relation_query = '''
                MATCH (n:TextChunk {id: $node_id})-[r:SIMILAR_TO]-(related:TextChunk)
                WHERE related.id <> $node_id
                RETURN related.id as id, related.text as text, related.source as source, 
                       r.weight as relation_score, 'TextChunk' as node_type,
                       COALESCE(r.reinforcement_count, 0) as reinforcements
                ORDER BY r.weight DESC, reinforcements DESC
                LIMIT 3
                '''
            
            with self.driver.session() as session:
                related_result = session.run(relation_query, node_id=node_id)
                
                for related_record in related_result:
                    related_id = related_record['id']
                    if related_id not in processed_ids and len(expanded_results) < top_k:
                        # Bonus za wzmocnienia w uczeniu się
                        reinforcement_bonus = min(0.1, related_record['reinforcements'] * 0.02)
                        adjusted_score = related_record['relation_score'] * 0.8 + reinforcement_bonus
                        
                        expanded_results.append({
                            'data': {
                                'id': related_id,
                                'text': related_record['text'],
                                'source': related_record['source']
                            },
                            'score': adjusted_score,
                            'node_type': related_record['node_type'],
                            'relation': 'SIMILAR_TO',
                            'reinforcements': related_record['reinforcements']
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
    Enhanced with learning-aware maintenance.
    """
    def __init__(self, connector: Neo4jConnector, similarity_threshold: float = 0.7):
        self.connector = connector
        self.text_graph = TextGraphBuilder(connector, similarity_threshold)
    
    def run_maintenance(self):
        """Uruchamia konserwację grafu z funkcjami uczenia się."""
        self.text_graph.run_maintenance()
    
    def get_learning_statistics(self):
        """Pobiera statystyki uczenia się grafu."""
        return self.text_graph.analyze_learning_patterns()
    
    def close(self):
        """Zamyka połączenie z bazą danych."""
        self.text_graph.close()


class LearningPatternTracker:
    """
    Nowa klasa do śledzenia wzorców uczenia się i dostosowywania grafu.
    """
    def __init__(self, connector: Neo4jConnector):
        self.driver = connector.get_driver()
        self.query_history = []
    
    def close(self):
        self.driver.close()
    
    def record_query_session(self, query: str, retrieved_nodes: list, user_feedback: dict = None):
        """
        Zapisuje sesję zapytania wraz z pobranymi węzłami i opcjonalnym feedbackiem użytkownika.
        """
        session_data = {
            'timestamp': int(time.time()),
            'query': query,
            'retrieved_nodes': retrieved_nodes,
            'feedback': user_feedback or {}
        }
        self.query_history.append(session_data)
        
        # Jeśli użytkownik wskazał przydatne wyniki, wzmocnij te relacje
        if user_feedback:
            self._process_user_feedback(retrieved_nodes, user_feedback)
    
    def _process_user_feedback(self, retrieved_nodes: list, feedback: dict):
        """
        Przetwarza feedback użytkownika i wzmacnia odpowiednie relacje.
        """
        # Przykład: feedback = {'useful': [node_id1, node_id2], 'not_useful': [node_id3]}
        useful_nodes = feedback.get('useful', [])
        not_useful_nodes = feedback.get('not_useful', [])
        
        # Wzmocnij relacje między użytecznymi węzłami
        for i, node_a in enumerate(useful_nodes):
            for node_b in useful_nodes[i+1:]:
                query = '''
                MATCH (a:Chunk {id: $node_a}), (b:Chunk {id: $node_b})
                MERGE (a)-[r:SIMILAR_TO]-(b)
                ON CREATE SET r.weight = 0.4, 
                             r.user_reinforced = true, 
                             r.last_used = timestamp(),
                             r.reinforcement_count = 1
                ON MATCH SET r.weight = CASE 
                               WHEN r.weight + 0.3 > 1.0 THEN 1.0
                               ELSE r.weight + 0.3
                             END,
                             r.user_reinforced = true, 
                             r.last_used = timestamp(),
                             r.reinforcement_count = COALESCE(r.reinforcement_count, 0) + 1
                '''
                with self.driver.session() as session:
                    session.run(query, node_a=node_a, node_b=node_b)
        
        # Osłab relacje do nieużytecznych węzłów
        for useful_node in useful_nodes:
            for not_useful_node in not_useful_nodes:
                query = '''
                MATCH (a:Chunk {id: $useful})-[r:SIMILAR_TO]-(b:Chunk {id: $not_useful})
                SET r.weight = CASE 
                               WHEN r.weight - 0.2 < 0.1 THEN 0.1
                               ELSE r.weight - 0.2
                             END,
                    r.user_downgraded = true
                '''
                with self.driver.session() as session:
                    session.run(query, useful=useful_node, not_useful=not_useful_node)
    
    def analyze_usage_patterns(self):
        """
        Analizuje wzorce użycia i identyfikuje trendy.
        """
        # Znajdź najczęściej używane węzły
        query = '''
        MATCH (n:Chunk)
        WHERE n.usage_count IS NOT NULL AND n.type = 'text'
        RETURN n.id as id, n.usage_count as usage, n.type as type,
               size(n.contexts) as context_variety
        ORDER BY n.usage_count DESC
        LIMIT 10
        '''
        
        with self.driver.session() as session:
            result = session.run(query)
            popular_nodes = [dict(record) for record in result]
        
        # Znajdź najsilniejsze relacje
        strong_relations_query = '''
        MATCH (a:Chunk)-[r:SIMILAR_TO]-(b:Chunk)
        WHERE r.weight > 0.7
        RETURN a.id as node_a, b.id as node_b, r.weight as weight, 
               r.reinforcement_count as reinforcements,
               r.user_reinforced as user_reinforced
        ORDER BY r.weight DESC, r.reinforcement_count DESC
        LIMIT 10
        '''
        
        with self.driver.session() as session:
            result = session.run(strong_relations_query)
            strong_relations = [dict(record) for record in result]
        
        return {
            'popular_nodes': popular_nodes,
            'strong_relations': strong_relations,
            'total_queries': len(self.query_history),
            'feedback_sessions': len([s for s in self.query_history if s['feedback']])
        }