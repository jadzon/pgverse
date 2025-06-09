from neo4j import GraphDatabase, Config
import time
from collections import defaultdict


class Neo4jConnector:
    """
    Manages Neo4j driver connection with server version check disabled.
    """
    def __init__(self, uri: str, user: str, password: str, config: Config):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.config = config

    def get_driver(self):
        return self.driver

    def close(self):
        if self.driver:
            self.driver.close()


class GraphBuilder:
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
    def __init__(self, connector: Neo4jConnector, similarity_threshold: float = 0.95):
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

    def create_relations(self):
        """
        Create or update SIMILAR_TO relationships between nodes of the same type based on cosine similarity.
        BEZ LIMITÓW - przetwarza wszystkie węzły
        """
        node_types = ['TextNode', 'ImageNode', 'FormulaNode', 'TableNode']
        
        for node_type in node_types:
            print(f"🔄 Przetwarzanie typu węzła: {node_type}")
            
            # Najpierw policz ile węzłów tego typu mamy
            count_query = f'''
            MATCH (n:{node_type})
            WHERE n.embedding IS NOT NULL
            RETURN count(n) as node_count
            '''
            
            with self.driver.session() as session:
                result = session.run(count_query)
                node_count = result.single()['node_count']
                print(f"  📊 Znaleziono {node_count} węzłów typu {node_type} z embeddingami")
                
                if node_count == 0:
                    print(f"  ⚠️ Brak węzłów {node_type} - pomijam")
                    continue
                
                # INFORMACJA bez limitowania
                if node_count > 1000:
                    estimated_time = (node_count * node_count / 2) / 5000
                    print(f"  ⏰ Szacowany czas: {estimated_time:.1f} minut")
                    print(f"  🚀 Rozpoczynam przetwarzanie {node_count} węzłów...")
            
                # ZAWSZE używaj prostego podejścia dla wszystkich rozmiarów
                print(f"  🔄 Tworzenie relacji podobieństwa...")
                start_time = time.time()
                
                simple_query = f'''
                MATCH (a:{node_type}), (b:{node_type})
                WHERE elementId(a) < elementId(b)
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
                
                try:
                    result = session.run(simple_query, threshold=self.threshold)
                    summary = result.consume()
                    end_time = time.time()
                    
                    relations_created = summary.counters.relationships_created
                    duration = end_time - start_time
                    
                    print(f"  ✅ ZAKOŃCZONO {node_type}: Utworzono {relations_created} relacji w {duration:.1f}s")
                    
                except Exception as e:
                    print(f"  ❌ Błąd dla {node_type}: {e}")
                    import traceback
                    print(f"  🔍 Szczegóły błędu: {traceback.format_exc()}")
        
        print()

    def create_relations_with_progress_callback(self, progress_callback=None):
        """
        Wersja z callbackiem do GUI - ZOPTYMALIZOWANA dla dużej liczby węzłów
        """
        node_types = ['TextNode', 'ImageNode', 'FormulaNode', 'TableNode']
        total_types = len(node_types)
        
        for type_idx, node_type in enumerate(node_types):
            if progress_callback:
                progress_callback(f"🔄 Przetwarzanie {node_type} ({type_idx+1}/{total_types})")
        
            # Policz węzły
            count_query = f'MATCH (n:{node_type}) WHERE n.embedding IS NOT NULL RETURN count(n) as cnt'
            with self.driver.session() as session:
                node_count = session.run(count_query).single()['cnt']
                
                if progress_callback:
                    progress_callback(f"  📊 {node_type}: {node_count} węzłów")
                
                if node_count == 0:
                    if progress_callback:
                        progress_callback(f"  ⚠️ Brak węzłów {node_type} - pomijam")
                    continue
                
                # NOWE: Inteligentny wybór strategii na podstawie liczby węzłów
                if node_count <= 200:
                    # Małe zestawy - standardowe podejście
                    if progress_callback:
                        progress_callback(f"  🚀 Mały zestaw - standardowe przetwarzanie...")
                    self._create_relations_standard(node_type, session, progress_callback)
                
                elif node_count <= 1000:
                    # Średnie zestawy - przetwarzanie partiami
                    if progress_callback:
                        progress_callback(f"  ⚡ Średni zestaw - przetwarzanie partiami...")
                    self._create_relations_batched(node_type, session, progress_callback, batch_size=50)
                
                else:
                    # Duże zestawy - zaawansowane przetwarzanie z optymalizacjami
                    if progress_callback:
                        progress_callback(f"  🔥 Duży zestaw ({node_count}) - zaawansowane przetwarzanie...")
                        estimated_time = (node_count * node_count / 2) / 2000  # Lepsze oszacowanie
                        progress_callback(f"  ⏰ Szacowany czas: {estimated_time:.1f} minut")
                    
                    self._create_relations_optimized(node_type, session, progress_callback)

        if progress_callback:
            progress_callback("🎉 Wszystkie typy węzłów przetworzone!")

    def _create_relations_standard(self, node_type, session, progress_callback):
        """Standardowe tworzenie relacji dla małych zestawów"""
        start_time = time.time()
        
        query = f'''
        MATCH (a:{node_type}), (b:{node_type})
        WHERE elementId(a) < elementId(b)
          AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
          AND NOT (a)-[:SIMILAR_TO]-(b)
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
        CREATE (a)-[r:SIMILAR_TO]->(b)
        SET r.weight = sim, 
            r.last_used = timestamp(), 
            r.created_at = timestamp(),
            r.reinforcement_count = 0,
            r.relation_type = 'similarity'
        '''
        
        try:
            result = session.run(query, threshold=self.threshold)
            summary = result.consume()
            end_time = time.time()
            
            relations_created = summary.counters.relationships_created
            duration = end_time - start_time
            
            if progress_callback:
                progress_callback(f"  ✅ {node_type}: {relations_created} relacji w {duration:.1f}s")
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"  ❌ Błąd standardowy dla {node_type}: {e}")

    def _create_relations_batched(self, node_type, session, progress_callback, batch_size=50):
        """Przetwarzanie partiami dla średnich zestawów"""
        start_time = time.time()
        total_relations = 0
        
        # Pobierz węzły partiami
        offset = 0
        batch_num = 0
        
        while True:
            batch_num += 1
            
            # Pobierz partię węzłów
            batch_query = f'''
            MATCH (n:{node_type})
            WHERE n.embedding IS NOT NULL
            RETURN n.id as id, n.embedding as embedding
            SKIP $offset LIMIT $batch_size
            '''
            
            batch_result = session.run(batch_query, offset=offset, batch_size=batch_size)
            batch_nodes = [dict(record) for record in batch_result]
            
            if not batch_nodes:
                break
            
            if progress_callback:
                progress_callback(f"    📦 Partia {batch_num}: {len(batch_nodes)} węzłów (offset {offset})")
            
            # Dla każdego węzła w partii, porównaj z WSZYSTKIMI następnymi węzłami
            for i, node_a in enumerate(batch_nodes):
                # Porównaj z pozostałymi węzłami w tej partii
                for j in range(i + 1, len(batch_nodes)):
                    node_b = batch_nodes[j]
                    relations_count = self._create_single_relation_if_similar(
                        node_a, node_b, session
                    )
                    if relations_count is not None:
                        total_relations += relations_count
                
                # Porównaj z wszystkimi węzłami POZA tą partią (tylko te z wyższym offset)
                compare_query = f'''
                MATCH (a:{node_type} {{id: $node_a_id}}), (b:{node_type})
                WHERE elementId(a) < elementId(b)
                  AND b.embedding IS NOT NULL
                  AND NOT b.id IN $current_batch_ids
                  AND NOT (a)-[:SIMILAR_TO]-(b)
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
                CREATE (a)-[r:SIMILAR_TO]->(b)
                SET r.weight = sim, 
                    r.last_used = timestamp(), 
                    r.created_at = timestamp(),
                    r.reinforcement_count = 0,
                    r.relation_type = 'similarity'
                RETURN count(r) as relations_created
                '''
                
                current_batch_ids = [node['id'] for node in batch_nodes]
                try:
                    comp_result = session.run(compare_query, 
                                            node_a_id=node_a['id'], 
                                            threshold=self.threshold,
                                            current_batch_ids=current_batch_ids)
                    comp_record = comp_result.single()
                    relations_count = comp_record['relations_created'] if comp_record else 0
                    if relations_count is not None:
                        total_relations += relations_count
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"    ⚠️ Błąd porównywania węzła {node_a['id']}: {e}")
            
            offset += batch_size
            
            if progress_callback and batch_num % 5 == 0:
                progress_callback(f"    🔗 Po {batch_num} partiach: {total_relations} relacji")
        
        end_time = time.time()
        duration = end_time - start_time
        
        if progress_callback:
            progress_callback(f"  ✅ {node_type}: {total_relations} relacji w {duration:.1f}s (partie)")

    def _create_single_relation_if_similar(self, node_a, node_b, session):
        """Tworzy relację między dwoma węzłami jeśli są podobne i relacja jeszcze nie istnieje"""
        try:
            # Sprawdź czy węzły mają embeddingi
            if not node_a.get('embedding') or not node_b.get('embedding'):
                return 0
                
            # Sprawdź czy relacja już istnieje
            check_query = '''
            MATCH (a {id: $id_a}), (b {id: $id_b})
            RETURN EXISTS((a)-[:SIMILAR_TO]-(b)) as relation_exists
            '''
            
            check_result = session.run(check_query, id_a=node_a['id'], id_b=node_b['id'])
            if check_result.single()['relation_exists']:
                return 0  # Relacja już istnieje, nie twórz duplikatu
                
            # Oblicz podobieństwo lokalnie (szybciej niż w Cypher)
            emb_a = node_a['embedding']
            emb_b = node_b['embedding']
            
            # Sprawdź czy embeddingi mają odpowiednią długość
            if len(emb_a) == 0 or len(emb_b) == 0:
                return 0
            
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(emb_a, emb_b))
            norm_a = sum(a * a for a in emb_a) ** 0.5
            norm_b = sum(b * b for b in emb_b) ** 0.5
            
            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
                
                if similarity >= self.threshold:
                    # Utwórz relację używając CREATE (nie MERGE)
                    relation_query = '''
                    MATCH (a {id: $id_a}), (b {id: $id_b})
                    CREATE (a)-[r:SIMILAR_TO]->(b)
                    SET r.weight = $similarity, 
                        r.last_used = timestamp(), 
                        r.created_at = timestamp(),
                        r.reinforcement_count = 0,
                        r.relation_type = 'similarity'
                    RETURN count(r) as created
                    '''
                    
                    try:
                        result = session.run(relation_query, 
                                           id_a=node_a['id'], 
                                           id_b=node_b['id'], 
                                           similarity=similarity)
                        record = result.single()
                        return record['created'] if record else 0
                    except Exception:
                        return 0
        
        except Exception:
            return 0
        
        return 0

    def _create_relations_optimized(self, node_type, session, progress_callback):
        """Zoptymalizowane tworzenie relacji dla dużych zestawów"""
        start_time = time.time()
        
        if progress_callback:
            progress_callback(f"  🔧 Optymalizacja 1: Tworzenie indeksów...")
        
        # Utwórz indeksy dla wydajności
        try:
            session.run(f"CREATE INDEX {node_type}_embedding_idx IF NOT EXISTS FOR (n:{node_type}) ON (n.embedding)")
            session.run(f"CREATE INDEX {node_type}_id_idx IF NOT EXISTS FOR (n:{node_type}) ON (n.id)")
        except Exception:
            pass
        
        if progress_callback:
            progress_callback(f"  ⚡ Optymalizacja 2: Przetwarzanie z threshold {self.threshold}...")
        
        # ZMIANA: Usuń hardcoded threshold - użyj tylko self.threshold
        # high_threshold = max(self.threshold, 0.95)  # ← USUŃ TO
        used_threshold = self.threshold  # ← UŻYWAJ TEGO ZAWSZE

        optimized_query = f'''
        CALL {{
            MATCH (a:{node_type})
            WHERE a.embedding IS NOT NULL
            WITH a LIMIT 100
            MATCH (b:{node_type})
            WHERE b.embedding IS NOT NULL AND elementId(a) < elementId(b)
              AND NOT (a)-[:SIMILAR_TO]-(b)
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
            CREATE (a)-[r:SIMILAR_TO]->(b)
            SET r.weight = sim, 
                r.last_used = timestamp(), 
                r.created_at = timestamp(),
                r.reinforcement_count = 0,
                r.relation_type = 'similarity'
            RETURN count(r) as batch_relations
        }} IN TRANSACTIONS OF 50 ROWS
        '''
        
        total_relations = 0
        
        try:
            # Przetwarzaj w mniejszych grupach
            for offset in range(0, 1000, 100):  # Maksymalnie 1000 węzłów w grupach po 100
                if progress_callback:
                    progress_callback(f"    🔄 Grupa węzłów {offset}-{offset+100}...")
                
                batch_query = optimized_query.replace("LIMIT 100", f"SKIP {offset} LIMIT 100")
                # ZMIANA: Używaj used_threshold (czyli self.threshold)
                result = session.run(batch_query, threshold=used_threshold)
                
                # Policz wyniki z każdej transakcji
                for record in result:
                    total_relations += record['batch_relations']
                
                if progress_callback and offset % 300 == 0:
                    progress_callback(f"      📊 Dotychczas: {total_relations} relacji...")
            
            end_time = time.time()
            duration = end_time - start_time
            
            if progress_callback:
                progress_callback(f"  ✅ {node_type}: {total_relations} relacji w {duration:.1f}s (optymalizowane)")
                # ZMIANA: Pokaż używany threshold z ustawień
                progress_callback(f"  📋 Użyto threshold: {used_threshold} (z ustawień aplikacji)")
                
        except Exception as e:
            if progress_callback:
                progress_callback(f"  ❌ Błąd optymalizowany dla {node_type}: {e}")
                progress_callback(f"  🔄 Próba fallback z ograniczeniem...")
            
            try:
                fallback_query = f'''
                MATCH (a:{node_type}), (b:{node_type})
                WHERE elementId(a) < elementId(b)
                  AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
                  AND NOT (a)-[:SIMILAR_TO]-(b)
                WITH a, b LIMIT 125000
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
                CREATE (a)-[r:SIMILAR_TO]->(b)
                SET r.weight = sim, 
                    r.last_used = timestamp(), 
                    r.created_at = timestamp(),
                    r.reinforcement_count = 0,
                    r.relation_type = 'similarity'
                '''
                
                # ZMIANA: Używaj self.threshold również w fallback
                result = session.run(fallback_query, threshold=self.threshold)
                summary = result.consume()
                relations_created = summary.counters.relationships_created
                
                if progress_callback:
                    progress_callback(f"  ✅ Fallback {node_type}: {relations_created} relacji (ograniczony zestaw)")
                    progress_callback(f"  📋 Fallback threshold: {self.threshold}")
                    
            except Exception as e2:
                if progress_callback:
                    progress_callback(f"  💥 Krytyczny błąd {node_type}: {e2}")
    
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
        POPRAWIONA WERSJA - zawsze zwraca słownik
        """
        try:
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
                # POPRAWKA 1: Bezpieczne wykonywanie zapytań z obsługą błędów
                try:
                    node_stats = [dict(record) for record in session.run(node_stats_query)]
                except Exception as e:
                    print(f"DEBUG: Błąd node_stats_query: {e}")
                    node_stats = []
                
                try:
                    relation_stats = [dict(record) for record in session.run(relation_stats_query)]
                except Exception as e:
                    print(f"DEBUG: Błąd relation_stats_query: {e}")
                    relation_stats = []
                
                try:
                    source_stats = [dict(record) for record in session.run(source_stats_query)]
                except Exception as e:
                    print(f"DEBUG: Błąd source_stats_query: {e}")
                    source_stats = []
                
                # Ogólne statystyki
                try:
                    total_query = '''
                    MATCH (n) 
                    RETURN count(n) as total_nodes, 
                           count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) as nodes_with_embeddings,
                           count(CASE WHEN n.base64 IS NOT NULL THEN 1 END) as nodes_with_base64
                    '''
                    total_stats = session.run(total_query).single()
                    total_stats_dict = dict(total_stats) if total_stats else {}
                except Exception as e:
                    print(f"DEBUG: Błąd total_query: {e}")
                    total_stats_dict = {'total_nodes': 0, 'nodes_with_embeddings': 0, 'nodes_with_base64': 0}
                
                # POPRAWKA 2: ZAWSZE zwróć słownik, nigdy listę
                result_dict = {
                    'node_statistics_by_type': node_stats,
                    'relation_statistics': relation_stats,
                    'source_statistics': source_stats,
                    'total_statistics': total_stats_dict,
                    'usage_patterns_count': len(self.usage_patterns) if hasattr(self, 'usage_patterns') else 0,
                    'current_threshold': self.threshold
                }
                
                return result_dict
                
        except Exception as e:
            print(f"DEBUG: Krytyczny błąd w analyze_learning_patterns: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            
            # AWARYJNY słownik - NIGDY nie zwracaj listy
            return {
                'error': str(e),
                'node_statistics_by_type': [],
                'relation_statistics': [],
                'source_statistics': [],
                'total_statistics': {'total_nodes': 0, 'nodes_with_embeddings': 0, 'nodes_with_base64': 0},
                'usage_patterns_count': 0,
                'current_threshold': getattr(self, 'threshold', 0.85)
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
        """
        print("Rozpoczynam konserwację grafu...")
        
        # Zastosuj zanikanie relacji
        self.decay_relationships()
        
        # Dostosuj próg adaptacyjnie
        self.adaptive_threshold_adjustment()
        
        # Wyczyść stare relacje
        self.prune_old()
        
        print("Konserwacja grafu zakończona.")

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