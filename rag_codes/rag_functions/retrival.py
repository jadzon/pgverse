from typing import List, Dict, Any, Optional
import time
import numpy as np
from .graph import Neo4jConnector, GraphBuilder, HybridTextRetriever, LearningPatternTracker
from .embeddings import CLIPEmbedder
from .source_calculate import TextRetriever
class TextRetriever:
    """
    Specjalizowany retriever dla wyszukiwania tekstowego w Neo4j z obsługą:
    - Węzłów TextNode
    - Relacji SIMILAR_TO
    - Uczenia się wzorców użycia
    - Hybrydowego wyszukiwania tekstowego
    - Feedbacku użytkownika
    - Filtrowania według źródeł
    """
    
    def __init__(self, connector: Neo4jConnector, similarity_threshold: float = 0.85):
        """
        Inicjalizuje TextRetriever
        
        Args:
            connector: Połączenie z Neo4j
            similarity_threshold: Próg podobieństwa dla relacji SIMILAR_TO
        """
        self.connector = connector
        self.driver = connector.get_driver()
        self.similarity_threshold = similarity_threshold
        
        # Inicjalizuj komponenty specyficzne dla tekstu
        self.graph_builder = GraphBuilder(connector, similarity_threshold)
        self.hybrid_retriever = HybridTextRetriever(connector)
        self.learning_tracker = LearningPatternTracker(connector)
        
        # Ustaw połączenie z graph_builder dla uczenia się
        self.hybrid_retriever.graph_builder = self.graph_builder
        
        # Embedder - używa singletona
        self.embedder = CLIPEmbedder.get_instance()
        
        # Konfiguracja specyficzna dla tekstu
        self.default_top_k = 10
        self.boost_similar_relations = True
        self.learning_enabled = True
        self.text_score_threshold = 0.7  # Wyższy próg dla tekstu
        
        print("✅ TextRetriever zainicjalizowany")
    
    def search(self, query: str, top_k: int = None, 
               score_threshold: float = None, 
               use_relations: bool = True,
               expand_with_similar: bool = True,
               source_filter: List[str] = None,
               subject_filter: List[str] = None) -> List[Dict[str, Any]]:
        """
        Główna metoda wyszukiwania tekstowego
        
        Args:
            query: Zapytanie tekstowe
            top_k: Liczba wyników do zwrócenia
            score_threshold: Minimalny próg podobieństwa (domyślnie 0.7)
            use_relations: Czy używać relacji SIMILAR_TO
            expand_with_similar: Czy rozszerzać wyniki o podobne węzły
            source_filter: Lista źródeł do filtrowania ['wikipedia', 'książka', ...]
            subject_filter: Lista przedmiotów do filtrowania ['matematyka', 'fizyka', ...]
            
        Returns:
            Lista wyników z metadanymi
        """
        if top_k is None:
            top_k = self.default_top_k
        if score_threshold is None:
            score_threshold = self.text_score_threshold
            
        print(f"📝 TextRetriever - wyszukiwanie: '{query}' (top_k={top_k}, threshold={score_threshold})")
        
        # Generuj embedding dla zapytania tekstowego
        query_embedding = self.embedder.get_text_embedding(query)
        if query_embedding is None:
            print("❌ Nie udało się wygenerować embeddingu dla zapytania tekstowego")
            return []
        
        # Podstawowe wyszukiwanie tekstowe
        initial_results = self._search_text_nodes(
            query_embedding, 
            top_k=top_k * 2 if expand_with_similar else top_k,
            score_threshold=score_threshold,
            source_filter=source_filter,
            subject_filter=subject_filter
        )
        
        if not initial_results:
            print("❌ Brak wyników z podstawowego wyszukiwania tekstowego")
            return []
        
        print(f"📋 Podstawowe wyszukiwanie tekstowe: {len(initial_results)} wyników")
        
        # Rozszerz wyniki używając relacji SIMILAR_TO
        if expand_with_similar and use_relations:
            expanded_results = self._expand_text_with_similar_relations(
                initial_results, 
                query_embedding,
                max_expansion=top_k,
                source_filter=source_filter,
                subject_filter=subject_filter
            )
            print(f"🔗 Po rozszerzeniu tekstowymi relacjami: {len(expanded_results)} wyników")
        else:
            expanded_results = initial_results
        
        # Sortuj i ogranicz wyniki
        final_results = self._rank_and_limit_text_results(expanded_results, top_k)
        
        # Zapisz wzorce użycia dla uczenia się
        if self.learning_enabled:
            retrieved_node_ids = [result['data']['id'] for result in final_results]
            self.graph_builder.track_usage_pattern(retrieved_node_ids, query[:100])
            self.learning_tracker.record_query_session(query, retrieved_node_ids)
        
        print(f"✅ Finalne wyniki tekstowe: {len(final_results)} dokumentów")
        return final_results
    
    def _search_text_nodes(self, query_embedding: List[float], 
                          top_k: int = 10, 
                          score_threshold: float = 0.7,
                          source_filter: List[str] = None,
                          subject_filter: List[str] = None) -> List[Dict[str, Any]]:
        """
        Wyszukiwanie tylko w węzłach TextNode używając cosine similarity
        """
        # Buduj klauzulę WHERE z filtrami
        where_conditions = ["n:TextNode", "n.embedding IS NOT NULL"]
        
        if source_filter:
            source_filter_str = "', '".join(source_filter)
            where_conditions.append(f"n.source IN ['{source_filter_str}']")
        
        if subject_filter:
            # Filtruj według przedmiotów w ścieżce
            subject_conditions = []
            for subject in subject_filter:
                subject_conditions.append(f"n.path CONTAINS '{subject}'")
            where_conditions.append(f"({' OR '.join(subject_conditions)})")
        
        where_clause = " AND ".join(where_conditions)
        
        search_query = f'''
        MATCH (n)
        WHERE {where_clause}
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
               n.type as data_type, n.base64 as base64_data, 
               n.usage_count as usage_count, score
        ORDER BY score DESC
        LIMIT $top_k
        '''
        
        with self.driver.session() as session:
            result = session.run(search_query, 
                               query_emb=query_embedding.tolist(), 
                               threshold=score_threshold, 
                               top_k=top_k)
            
            results = []
            for record in result:
                results.append({
                    'data': {
                        'id': record['id'],
                        'text': record['text'],
                        'path': record['path'],
                        'source': record['source'] or 'unknown',
                        'type': record['data_type'],
                        'base64': record['base64_data'],
                        'usage_count': record['usage_count'] or 0
                    },
                    'score': record['score'],
                    'source_type': record['source'] or 'unknown',
                    'retrieval_method': 'text_similarity',
                    'node_type': 'TextNode'
                })
            
            return results
    
    def _expand_text_with_similar_relations(self, initial_results: List[Dict[str, Any]], 
                                          query_embedding: List[float],
                                          max_expansion: int = 10,
                                          source_filter: List[str] = None,
                                          subject_filter: List[str] = None) -> List[Dict[str, Any]]:
        """
        Rozszerza wyniki tekstowe używając relacji SIMILAR_TO - tylko TextNode
        """
        if not initial_results:
            return initial_results
        
        # Pobierz ID węzłów z początkowych wyników
        initial_node_ids = [result['data']['id'] for result in initial_results]
        
        # Buduj filtry dla podobnych węzłów
        filter_conditions = ["similar:TextNode", "similar.embedding IS NOT NULL", "NOT similar.id IN $initial_ids"]
        
        if source_filter:
            source_filter_str = "', '".join(source_filter)
            filter_conditions.append(f"similar.source IN ['{source_filter_str}']")
        
        if subject_filter:
            subject_conditions = []
            for subject in subject_filter:
                subject_conditions.append(f"similar.path CONTAINS '{subject}'")
            filter_conditions.append(f"({' OR '.join(subject_conditions)})")
        
        similar_filter = " AND ".join(filter_conditions)
        
        # Znajdź podobne węzły tekstowe przez relacje SIMILAR_TO
        similar_query = f'''
        MATCH (initial)
        WHERE initial.id IN $initial_ids
        MATCH (initial)-[r:SIMILAR_TO]-(similar)
        WHERE {similar_filter}
        WITH DISTINCT similar, 
             max(r.weight) as max_relation_weight,
             count(r) as relation_count,
             reduce(dot = 0.0, i IN range(0, size(similar.embedding)-1) |
                 dot + similar.embedding[i] * $query_emb[i]
             ) /
             (
                 sqrt(reduce(norm_n = 0.0, i IN range(0, size(similar.embedding)-1) |
                     norm_n + similar.embedding[i] * similar.embedding[i]
                 )) *
                 sqrt(reduce(norm_q = 0.0, i IN range(0, size($query_emb)-1) |
                     norm_q + $query_emb[i] * $query_emb[i]
                 ))
             ) AS direct_score
        RETURN similar.id as id, similar.text as text, similar.path as path, 
               similar.source as source, similar.type as data_type, 
               similar.base64 as base64_data, similar.usage_count as usage_count,
               direct_score, max_relation_weight, relation_count,
               (direct_score * 0.7 + max_relation_weight * 0.3) as combined_score
        ORDER BY combined_score DESC
        LIMIT $max_expansion
        '''
        
        with self.driver.session() as session:
            result = session.run(similar_query,
                               initial_ids=initial_node_ids,
                               query_emb=query_embedding.tolist(),
                               max_expansion=max_expansion)
            
            similar_results = []
            for record in result:
                similar_results.append({
                    'data': {
                        'id': record['id'],
                        'text': record['text'],
                        'path': record['path'],
                        'source': record['source'] or 'unknown',
                        'type': record['data_type'],
                        'base64': record['base64_data'],
                        'usage_count': record['usage_count'] or 0
                    },
                    'score': record['combined_score'],
                    'direct_score': record['direct_score'],
                    'relation_weight': record['max_relation_weight'],
                    'relation_count': record['relation_count'],
                    'source_type': record['source'] or 'unknown',
                    'retrieval_method': 'text_similar_relation',
                    'node_type': 'TextNode'
                })
        
        # Połącz wyniki
        all_results = initial_results + similar_results
        
        # Usuń duplikaty na podstawie ID
        seen_ids = set()
        unique_results = []
        for result in all_results:
            node_id = result['data']['id']
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_and_limit_text_results(self, results: List[Dict[str, Any]], 
                                    top_k: int) -> List[Dict[str, Any]]:
        """
        Sortuje wyniki tekstowe według score z dodatkowymi czynnikami dla tekstu
        """
        # Dodaj dodatkowe czynniki rankingu specyficzne dla tekstu
        for result in results:
            base_score = result['score']
            usage_bonus = min(0.1, (result['data']['usage_count'] / 100) * 0.1)
            
            # Bonus za długość tekstu (dłuższe teksty mogą być bardziej wartościowe)
            text_length = len(result['data']['text']) if result['data']['text'] else 0
            length_bonus = min(0.05, (text_length / 1000) * 0.05)
            
            # Bonus za typ relacji
            method_bonus = 0.0
            if result.get('retrieval_method') == 'text_similarity':
                method_bonus = 0.05  # Bonus za bezpośrednie dopasowanie tekstowe
            elif result.get('retrieval_method') == 'text_similar_relation':
                relation_weight = result.get('relation_weight', 0)
                method_bonus = relation_weight * 0.1
            
            # Bonus za źródło (niektóre źródła mogą być bardziej wiarygodne)
            source_bonus = 0.0
            source = result['data']['source']
            if source in ['książka', 'artykuł_naukowy']:
                source_bonus = 0.03
            elif source in ['wikipedia']:
                source_bonus = 0.02
            
            # Oblicz finalne score dla tekstu
            result['final_score'] = base_score + usage_bonus + length_bonus + method_bonus + source_bonus
            result['ranking_details'] = {
                'base_score': base_score,
                'usage_bonus': usage_bonus,
                'length_bonus': length_bonus,
                'method_bonus': method_bonus,
                'source_bonus': source_bonus,
                'final_score': result['final_score'],
                'text_length': text_length
            }
        
        # Sortuj według final_score
        sorted_results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        return sorted_results[:top_k]
    
    def search_by_keywords(self, keywords: List[str], top_k: int = None,
                          logical_operator: str = "OR") -> List[Dict[str, Any]]:
        """
        Wyszukiwanie po słowach kluczowych w tekście węzłów
        
        Args:
            keywords: Lista słów kluczowych
            top_k: Liczba wyników
            logical_operator: "AND" lub "OR" dla łączenia słów kluczowych
        """
        if top_k is None:
            top_k = self.default_top_k
            
        print(f"🔍 Wyszukiwanie po słowach kluczowych: {keywords} ({logical_operator})")
        
        # Buduj warunki wyszukiwania
        if logical_operator.upper() == "AND":
            text_conditions = [f"toLower(n.text) CONTAINS toLower('{keyword}')" for keyword in keywords]
            text_condition = " AND ".join(text_conditions)
        else:  # OR
            text_conditions = [f"toLower(n.text) CONTAINS toLower('{keyword}')" for keyword in keywords]
            text_condition = " OR ".join(text_conditions)
        
        keyword_query = f'''
        MATCH (n:TextNode)
        WHERE n.text IS NOT NULL AND ({text_condition})
        RETURN n.id as id, n.text as text, n.path as path, n.source as source,
               n.type as data_type, n.base64 as base64_data, n.usage_count as usage_count,
               1.0 as score
        ORDER BY n.usage_count DESC, length(n.text) DESC
        LIMIT $top_k
        '''
        
        with self.driver.session() as session:
            result = session.run(keyword_query, top_k=top_k)
            
            results = []
            for record in result:
                results.append({
                    'data': {
                        'id': record['id'],
                        'text': record['text'],
                        'path': record['path'],
                        'source': record['source'] or 'unknown',
                        'type': record['data_type'],
                        'base64': record['base64_data'],
                        'usage_count': record['usage_count'] or 0
                    },
                    'score': record['score'],
                    'source_type': record['source'] or 'unknown',
                    'retrieval_method': 'keyword_search',
                    'node_type': 'TextNode',
                    'keywords_found': keywords
                })
        
        print(f"📝 Znaleziono {len(results)} wyników dla słów kluczowych")
        return results
    
    def get_text_context(self, node_id: str, context_size: int = 3) -> List[Dict[str, Any]]:
        """
        Pobiera kontekst tekstowy wokół określonego węzła
        
        Args:
            node_id: ID węzła
            context_size: Liczba podobnych węzłów do pobrania jako kontekst
        """
        return self.get_related_text_nodes(node_id, max_depth=2, top_k=context_size)
    
    def get_related_text_nodes(self, node_id: str, relation_types: List[str] = None,
                              max_depth: int = 2, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Pobiera powiązane węzły tekstowe przez relacje
        """
        if relation_types is None:
            relation_types = ['SIMILAR_TO']
        
        relation_filter = '|'.join(relation_types)
        
        query = f'''
        MATCH path = (start {{id: $node_id}})-[r:{relation_filter}*1..{max_depth}]-(related:TextNode)
        WHERE related.id <> $node_id
        WITH related, path,
             reduce(weight = 1.0, rel in relationships(path) | weight * rel.weight) as path_weight,
             length(path) as path_length
        RETURN DISTINCT related.id as id, related.text as text, related.path as path,
               related.source as source, related.type as data_type, related.base64 as base64_data,
               path_weight, path_length,
               path_weight / path_length as normalized_weight
        ORDER BY normalized_weight DESC
        LIMIT $top_k
        '''
        
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id, top_k=top_k)
            
            related_nodes = []
            for record in result:
                related_nodes.append({
                    'data': {
                        'id': record['id'],
                        'text': record['text'],
                        'path': record['path'],
                        'source': record['source'] or 'unknown',
                        'type': record['data_type'],
                        'base64': record['base64_data']
                    },
                    'path_weight': record['path_weight'],
                    'path_length': record['path_length'],
                    'score': record['normalized_weight'],
                    'retrieval_method': 'text_graph_traversal',
                    'node_type': 'TextNode'
                })
            
            return related_nodes
    
    def provide_feedback(self, query: str, retrieved_results: List[Dict[str, Any]], 
                        useful_ids: List[str], not_useful_ids: List[str]):
        """
        Dostarcza feedback użytkownika dla poprawy przyszłych wyników tekstowych
        """
        if not self.learning_enabled:
            return
            
        print(f"📝 TextRetriever feedback: {len(useful_ids)} użytecznych, {len(not_useful_ids)} nieużytecznych")
        
        feedback = {
            'useful': useful_ids,
            'not_useful': not_useful_ids,
            'total_results': len(retrieved_results),
            'search_type': 'text'
        }
        
        # Zapisz feedback w learning tracker
        retrieved_node_ids = [result['data']['id'] for result in retrieved_results]
        self.learning_tracker.record_query_session(query, retrieved_node_ids, feedback)
        
        # Wzmocnij relacje między użytecznymi węzłami tekstowymi
        if len(useful_ids) > 1:
            self.graph_builder.track_usage_pattern(useful_ids, f"text_feedback:{query[:50]}")
        
        print("✅ TextRetriever feedback zapisany")
    
    def get_text_statistics(self) -> Dict[str, Any]:
        """
        Zwraca statystyki specyficzne dla wyszukiwania tekstowego
        """
        with self.driver.session() as session:
            # Statystyki węzłów tekstowych
            text_stats = session.run('''
            MATCH (n:TextNode)
            RETURN 
                count(n) as total_text_nodes,
                count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) as text_nodes_with_embeddings,
                avg(length(n.text)) as avg_text_length,
                max(length(n.text)) as max_text_length,
                min(length(n.text)) as min_text_length
            ''').single()
            
            # Statystyki relacji tekstowych
            relation_stats = session.run('''
            MATCH (n1:TextNode)-[r:SIMILAR_TO]-(n2:TextNode)
            RETURN 
                count(r) as text_relations,
                avg(r.weight) as avg_relation_weight,
                max(r.weight) as max_relation_weight,
                min(r.weight) as min_relation_weight
            ''').single()
            
            # Statystyki źródeł
            source_stats = session.run('''
            MATCH (n:TextNode)
            WHERE n.source IS NOT NULL
            RETURN n.source as source, count(n) as count
            ORDER BY count DESC
            ''')
            
            source_distribution = {record['source']: record['count'] for record in source_stats}
        
        return {
            'text_nodes': {
                'total': text_stats['total_text_nodes'],
                'with_embeddings': text_stats['text_nodes_with_embeddings'],
                'avg_length': text_stats['avg_text_length'],
                'max_length': text_stats['max_text_length'],
                'min_length': text_stats['min_text_length']
            },
            'text_relations': {
                'total': relation_stats['text_relations'],
                'avg_weight': relation_stats['avg_relation_weight'],
                'max_weight': relation_stats['max_relation_weight'],
                'min_weight': relation_stats['min_relation_weight']
            },
            'source_distribution': source_distribution,
            'retriever_config': {
                'similarity_threshold': self.similarity_threshold,
                'text_score_threshold': self.text_score_threshold,
                'default_top_k': self.default_top_k,
                'learning_enabled': self.learning_enabled
            }
        }
    
    def close(self):
        """
        Zamyka wszystkie połączenia TextRetriever
        """
        print("🔒 Zamykanie TextRetriever...")
        
        if hasattr(self, 'hybrid_retriever'):
            self.hybrid_retriever.close()
        
        if hasattr(self, 'learning_tracker'):
            self.learning_tracker.close()
        
        if hasattr(self, 'graph_builder'):
            self.graph_builder.close()
        
        print("✅ TextRetriever zamknięty")

# Przykład użycia
if __name__ == "__main__":
    from .graph import Neo4jConnector
    
    # Inicjalizacja
    connector = Neo4jConnector("bolt://localhost:7687", "neo4j", "password")
    text_retriever = TextRetriever(connector)
    
    try:
        # Podstawowe wyszukiwanie tekstowe
        results = text_retriever.search("matematyka równania", top_k=5)
        print(f"Znaleziono {len(results)} wyników tekstowych")
        
        # Wyszukiwanie z filtrami
        filtered_results = text_retriever.search(
            "fizyka", 
            top_k=10,
            source_filter=['książka', 'artykuł_naukowy'],
            subject_filter=['fizyka']
        )
        print(f"Znaleziono {len(filtered_results)} filtrowanych wyników")
        
        # Wyszukiwanie słowami kluczowymi
        keyword_results = text_retriever.search_by_keywords(
            ['równanie', 'matematyka'], 
            logical_operator="AND"
        )
        print(f"Znaleziono {len(keyword_results)} wyników po słowach kluczowych")
        
        # Statystyki
        stats = text_retriever.get_text_statistics()
        print(f"Statystyki: {stats['text_nodes']['total']} węzłów tekstowych")
        
    finally:
        text_retriever.close()
        connector.close()