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
      - type: data type, e.g. 'text', 'image', 'table', 'formula'
      - text: textual content or metadata description
      - path: optional path to external data (e.g., image file, table CSV)
      - embedding: vector representation for similarity calculations
      - source: source type from sources_config.json
      - base64: base64 encoded data for images, formulas, tables
    Self-learning: reinforces edges when nodes are accessed together.
    """
    def __init__(self, connector: Neo4jConnector, similarity_threshold: float = 0.8):
        self.driver = connector.get_driver()
        self.threshold = similarity_threshold
        self.usage_patterns = defaultdict(int)
        self.last_threshold_adjustment = time.time()

    def close(self):
        self.driver.close()

    def insert_node(self, node_id: str, data_type: str, text: str, embedding: list = None, 
                   path: str = None, source: str = "unknown", base64_data: str = None):
        """
        Inserts or updates a Chunk node with given properties, including optional embedding, 
        source type, and base64 data for images/formulas/tables.
        
        Args:
            node_id: Unique identifier for the node
            data_type: Type of data ('text', 'image', 'formula', 'table')
            text: Text content or description
            embedding: Vector embedding (text or image embedding)
            path: File path to the original data
            source: Source type from sources_config.json
            base64_data: Base64 encoded data (only for image/formula/table)
        """
        # Determine node label based on data type
        node_label = self._get_node_label(data_type)
        
        # Base properties for all nodes
        node_properties = {
            'id': node_id,
            'type': data_type,
            'text': text,
            'embedding': embedding,
            'path': path,
            'source': source,
            'usage_count': 0,
            'created_at': int(time.time() * 1000),  # timestamp in ms
            'last_accessed': int(time.time() * 1000),
            'embedding_type': 'image_embedding' if data_type in ['image', 'formula', 'table'] else 'text_embedding'
        }
        
        # Add base64 data only for visual content
        if base64_data and data_type in ['image', 'formula', 'table']:
            node_properties['base64'] = base64_data
        
        # Create the query with appropriate label
        query = f'''
        MERGE (n:{node_label} {{id: $id}})
        SET n.type = $type,
            n.text = $text,
            n.embedding = $embedding,
            n.path = $path,
            n.source = $source,
            n.usage_count = COALESCE(n.usage_count, 0),
            n.created_at = COALESCE(n.created_at, $created_at),
            n.last_accessed = $last_accessed,
            n.embedding_type = $embedding_type
        '''
        
        # Add base64 property conditionally
        if base64_data and data_type in ['image', 'formula', 'table']:
            query += ', n.base64 = $base64'
        
        with self.driver.session() as session:
            session.run(query, **node_properties)

    def _get_node_label(self, data_type: str) -> str:
        """
        Returns appropriate node label based on data type.
        """
        label_mapping = {
            'text': 'TextNode',
            'image': 'ImageNode',
            'formula': 'FormulaNode', 
            'table': 'TableNode'
        }
        return label_mapping.get(data_type, 'Chunk')

    def get_image_embedding(self, image_path: str):
        """
        Gets image embedding using CLIP embedder.
        This should be called from DataBaseApp with proper CLIPEmbedder instance.
        """
        # This will be handled in DataBaseApp.py with CLIPEmbedder
        pass

    def create_text_relations(self):
        """
        Create or update SIMILAR_TO relationships between nodes based on cosine similarity.
        Now supports both text and image embeddings from CLIP.
        """
        # Query for all nodes that have embeddings (both text and image)
        query = '''
        MATCH (a), (b)
        WHERE (a:TextNode OR a:ImageNode OR a:FormulaNode OR a:TableNode) 
          AND (b:TextNode OR b:ImageNode OR b:FormulaNode OR b:TableNode)
          AND elementId(a) < elementId(b)
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
        ON CREATE SET r.weight = sim, 
                     r.last_used = timestamp(), 
                     r.reinforcement_count = 0, 
                     r.created_at = timestamp(),
                     r.relation_type = 'similarity'
        ON MATCH SET r.weight = sim, 
                    r.last_used = timestamp()
        '''
        with self.driver.session() as session:
            result = session.run(query, threshold=self.threshold)
            summary = result.consume()
            print(f"Utworzono/zaktualizowano {summary.counters.relationships_created} relacji podobieństwa")

    def create_multimodal_relations(self):
        """
        Creates enhanced relationships between different types of content.
        CLIP embeddings allow comparison between text and images.
        """
        # Relationships between text nodes and image/formula/table nodes
        multimodal_query = '''
        MATCH (t:TextNode), (v)
        WHERE (v:ImageNode OR v:FormulaNode OR v:TableNode)
          AND t.embedding IS NOT NULL AND v.embedding IS NOT NULL
          AND t.source = v.source  // Same source increases relevance
        WITH t, v,
        reduce(dot = 0.0, i IN range(0, size(t.embedding)-1) |
            dot + t.embedding[i] * v.embedding[i]
        ) /
        (
            sqrt(reduce(na = 0.0, i IN range(0, size(t.embedding)-1) |
                na + t.embedding[i] * t.embedding[i]
            )) *
            sqrt(reduce(nb = 0.0, i IN range(0, size(v.embedding)-1) |
                nb + v.embedding[i] * v.embedding[i]
            ))
        ) AS sim
        WHERE sim >= $threshold * 0.8  // Slightly lower threshold for cross-modal
        MERGE (t)-[r:RELATES_TO]->(v)
        ON CREATE SET r.weight = sim, 
                     r.last_used = timestamp(),
                     r.relation_type = 'cross_modal',
                     r.created_at = timestamp()
        ON MATCH SET r.weight = sim, 
                    r.last_used = timestamp()
        '''
        
        with self.driver.session() as session:
            result = session.run(multimodal_query, threshold=self.threshold)
            summary = result.consume()
            print(f"Utworzono/zaktualizowano {summary.counters.relationships_created} relacji cross-modal")

    def reinforce_relationship(self, node_a_id: str, node_b_id: str, strength: float = 1.0):
        """
        Wzmacnia relację między dwoma węzłami na podstawie ich współwystępowania.
        Jeśli relacja nie istnieje, tworzy ją z małą wagą.
        """
        query = '''
        MATCH (a {id: $node_a}), (b {id: $node_b})
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
        MATCH (n {id: $node_id})
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
            session.run(query, node_id=node_id, context=context[:100])

    def analyze_learning_patterns(self):
        """
        Analizuje wzorce uczenia się i zwraca rozszerzone statystyki.
        """
        # Statystyki węzłów według typu
        node_stats_query = '''
        MATCH (n)
        WHERE n.type IS NOT NULL
        RETURN n.type as data_type, count(n) as count, 
               avg(n.usage_count) as avg_usage,
               count(CASE WHEN n.base64 IS NOT NULL THEN 1 END) as with_base64,
               collect(DISTINCT n.source)[0..5] as sample_sources
        ORDER BY count DESC
        '''
        
        # Statystyki relacji
        relation_stats_query = '''
        MATCH ()-[r]->()
        RETURN type(r) as relation_type, count(r) as count,
               avg(r.weight) as avg_weight,
               max(r.weight) as max_weight,
               min(r.weight) as min_weight
        '''
        
        # Statystyki źródeł
        source_stats_query = '''
        MATCH (n)
        WHERE n.source IS NOT NULL
        RETURN n.source as source, count(n) as node_count,
               collect(DISTINCT n.type) as data_types
        ORDER BY node_count DESC
        '''
        
        with self.driver.session() as session:
            node_stats = [dict(record) for record in session.run(node_stats_query)]
            relation_stats = [dict(record) for record in session.run(relation_stats_query)]
            source_stats = [dict(record) for record in session.run(source_stats_query)]
            
            # Ogólne statystyki
            total_query = '''
            MATCH (n) 
            RETURN count(n) as total_nodes, 
                   count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) as nodes_with_embeddings,
                   count(CASE WHEN n.base64 IS NOT NULL THEN 1 END) as nodes_with_base64
            '''
            total_stats = session.run(total_query).single()
            
            return {
                'node_statistics_by_type': node_stats,
                'relation_statistics': relation_stats,
                'source_statistics': source_stats,
                'total_statistics': dict(total_stats),
                'usage_patterns_count': len(self.usage_patterns),
                'current_threshold': self.threshold
            }

    def decay_relationships(self, decay_factor: float = 0.95, min_weight: float = 0.1):
        """
        Zmniejsza wagę relacji, które nie były używane przez długi czas.
        Usuwa relacje poniżej minimalnej wagi.
        """
        thirty_days_ms = 30 * 24 * 60 * 60 * 1000
        seven_days_ms = 7 * 24 * 60 * 60 * 1000
        
        # Silny decay dla bardzo starych relacji
        strong_decay_query = '''
        MATCH ()-[r]->()
        WHERE r.last_used < timestamp() - $long_threshold
        SET r.weight = r.weight * $strong_decay,
            r.decay_applied = COALESCE(r.decay_applied, 0) + 1
        '''
        
        # Łagodny decay dla średnio starych relacji
        mild_decay_query = '''
        MATCH ()-[r]->()
        WHERE r.last_used < timestamp() - $short_threshold 
          AND r.last_used >= timestamp() - $long_threshold
        SET r.weight = r.weight * $mild_decay,
            r.decay_applied = COALESCE(r.decay_applied, 0) + 1
        '''
        
        # Usuń bardzo słabe relacje
        cleanup_query = '''
        MATCH ()-[r]->()
        WHERE r.weight < $min_weight AND COALESCE(r.reinforcement_count, 0) < 2
        DELETE r
        '''
        
        with self.driver.session() as session:
            # Zastosuj silny decay
            session.run(strong_decay_query, 
                       long_threshold=thirty_days_ms, 
                       strong_decay=decay_factor * 0.8)
            
            # Zastosuj łagodny decay  
            session.run(mild_decay_query,
                       short_threshold=seven_days_ms,
                       long_threshold=thirty_days_ms,
                       mild_decay=decay_factor)
            
            # Wyczyść słabe relacje
            result = session.run(cleanup_query, min_weight=min_weight)
            deleted_count = result.consume().counters.relationships_deleted
            
            print(f"Zastosowano decay relacji. Usunięto {deleted_count} słabych relacji.")

    def adaptive_threshold_adjustment(self):
        """
        Automatycznie dostosowuje próg podobieństwa na podstawie gęstości grafu.
        """
        current_time = time.time()
        if current_time - self.last_threshold_adjustment < 3600:  # 1 godzina
            return
            
        # Policz obecną liczbę relacji i węzłów
        stats_query = '''
        MATCH (n) 
        WHERE n.embedding IS NOT NULL
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE r.weight IS NOT NULL
        RETURN count(DISTINCT n) as node_count, count(r) as relation_count,
               avg(r.weight) as avg_weight, max(r.weight) as max_weight
        '''
        
        with self.driver.session() as session:
            result = session.run(stats_query).single()
            node_count = result['node_count']
            relation_count = result['relation_count']
            avg_weight = result['avg_weight'] or 0
            
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
                
                self.last_threshold_adjustment = current_time
                
                if abs(old_threshold - self.threshold) > 0.01:
                    print(f"Próg podobieństwa dostosowany: {old_threshold:.3f} → {self.threshold:.3f}")

    def prune_old(self, max_age_ms: int = 30*24*3600*1000):
        """
        Delete relationships not updated within max_age_ms.
        Enhanced with learning-aware pruning.
        """
        query = '''
        MATCH ()-[r]->()
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
        Enhanced for multimodal content.
        """
        print("Rozpoczynam konserwację grafu multimodalnego...")
        
        # Standardowe relacje podobieństwa
        self.create_text_relations()
        
        # Relacje cross-modalne (tekst-obraz)
        self.create_multimodal_relations()
        
        # Zastosuj zanikanie relacji
        self.decay_relationships()
        
        # Dostosuj próg adaptacyjnie
        self.adaptive_threshold_adjustment()
        
        # Wyczyść stare relacje
        self.prune_old()
        
        print("Konserwacja grafu multimodalnego zakończona.")


# Pozostałe klasy bez zmian...
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
        Enhanced with learning tracking and source information.
        """
        # Wyszukuj we wszystkich typach węzłów
        search_query = '''
        MATCH (n)
        WHERE (n:TextNode OR n:ImageNode OR n:FormulaNode OR n:TableNode) 
          AND n.embedding IS NOT NULL
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
        RETURN n.id as id, n.text as text, n.path as path, n.source as source, 
               n.type as data_type, n.base64 as base64_data, score
        ORDER BY score DESC
        LIMIT $top_k
        '''
        
        with self.driver.session() as session:
            result = session.run(search_query, query_emb=query_embedding, 
                               threshold=score_threshold, top_k=top_k)
            results = [{'data': {'id': record['id'], 'text': record['text'], 
                                'path': record['path'], 'source': record['source'] or 'unknown',
                                'type': record['data_type'], 'base64': record['base64_data']}, 
                       'score': record['score'], 'source_type': record['source'] or 'unknown'} 
                       for record in result]
        
        return results

    def search_by_image(self, query_embedding: list, top_k: int = 5, score_threshold: float = 0.66):
        """
        Wyszukuje najbardziej podobne obrazy na podstawie embeddingu zapytania.
        """
        query = '''
        MATCH (n)
        WHERE (n:ImageNode OR n:FormulaNode OR n:TableNode) 
          AND n.embedding IS NOT NULL
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
        RETURN n.id as id, n.text as description, n.path as image_path, 
               n.source as source, n.type as data_type, n.base64 as base64_data, score
        ORDER BY score DESC
        LIMIT $top_k
        '''
        
        with self.driver.session() as session:
            result = session.run(query, query_emb=query_embedding, 
                               threshold=score_threshold, top_k=top_k)
            return [{'data': {'id': record['id'], 'description': record['description'], 
                            'path': record['image_path'], 'source': record['source'] or 'unknown',
                            'type': record['data_type'], 'base64': record['base64_data']}, 
                    'score': record['score'], 'source_type': record['source'] or 'unknown'} 
                    for record in result]


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
        useful_nodes = feedback.get('useful', [])
        not_useful_nodes = feedback.get('not_useful', [])
        
        # Wzmocnij relacje między użytecznymi węzłami
        for i, node_a in enumerate(useful_nodes):
            for node_b in useful_nodes[i+1:]:
                query = '''
                MATCH (a {id: $node_a}), (b {id: $node_b})
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
    
    def analyze_usage_patterns(self):
        """
        Analizuje wzorce użycia i identyfikuje trendy.
        """
        # Znajdź najczęściej używane węzły
        query = '''
        MATCH (n)
        WHERE n.usage_count IS NOT NULL
        RETURN n.id as id, n.usage_count as usage, n.type as type,
               n.source as source, size(COALESCE(n.contexts, [])) as context_variety
        ORDER BY n.usage_count DESC
        LIMIT 10
        '''
        
        with self.driver.session() as session:
            result = session.run(query)
            popular_nodes = [dict(record) for record in result]
        
        return {
            'popular_nodes': popular_nodes,
            'total_queries': len(self.query_history),
            'feedback_sessions': len([s for s in self.query_history if s['feedback']])
        }