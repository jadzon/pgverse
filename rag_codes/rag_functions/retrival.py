from typing import List, Dict, Any, Optional
import time
import numpy as np
import re
from collections import defaultdict
from .graph import Neo4jConnector, LearningPatternTracker
from .embeddings import CLIPEmbedder
from source_calculate import HybridWeightFunction

class TextRetriever:
    """
    Standardowy retriever dla wyszukiwania tekstowego w Neo4j z obsługą:
    - Węzłów TextNode
    - Wyszukiwania semantycznego przez embeddingi
    - Wyszukiwania słowami kluczowymi
    - Filtrowania według źródeł i przedmiotów
    - Opcjonalnego automatycznego uczenia się wzorców
    """
    
    def __init__(self, connector: Neo4jConnector, similarity_threshold: float = 0.9, enable_learning: bool = False):
        """
        Inicjalizuje TextRetriever
        
        Args:
            connector: Połączenie z Neo4j
            similarity_threshold: Próg podobieństwa dla wyszukiwania
            enable_learning: Czy włączyć zaawansowane funkcje uczenia się
        """
        self.connector = connector
        self.driver = connector.get_driver()
        self.similarity_threshold = similarity_threshold
        self.enable_learning = enable_learning
        
        # Embedder - używa singletona
        self.embedder = CLIPEmbedder.get_instance()
        
        # Dodaj HybridWeightFunction
        self.hybrid_scorer = HybridWeightFunction(
            embedding_weight=0.7,
            source_weight=0.3
        )
        
        # Opcjonalnie dodaj LearningPatternTracker
        if self.enable_learning:
            self.learning_tracker = LearningPatternTracker(connector)
            print("✅ TextRetriever zainicjalizowany z zaawansowanym pattern tracking")
        else:
            self.learning_tracker = None
            print("✅ TextRetriever zainicjalizowany bez pattern tracking")
        
        # Konfiguracja
        self.default_top_k = 10
    
    def search(self, query: str, top_k: int = None, 
               score_threshold: float = None,
               source_filter: List[str] = None,
               subject_filter: List[str] = None,
               user_feedback: Dict[str, Any] = None,
               track_patterns: bool = None) -> List[Dict[str, Any]]:
        """
        Główna metoda wyszukiwania tekstowego z opcjonalnym śledzeniem wzorców
        
        Args:
            query: Zapytanie tekstowe
            top_k: Liczba wyników do zwrócenia
            score_threshold: Minimalny próg podobieństwa
            source_filter: Lista źródeł do filtrowania ['wikipedia', 'książka', ...]
            subject_filter: Lista przedmiotów do filtrowania ['matematyka', 'fizyka', ...]
            user_feedback: Feedback użytkownika z poprzednich zapytań
            track_patterns: Czy śledzić wzorce użycia dla uczenia się (domyślnie używa enable_learning)
            
        Returns:
            Lista wyników z metadanymi
        """
        if top_k is None:
            top_k = self.default_top_k
        if score_threshold is None:
            score_threshold = self.similarity_threshold
        if track_patterns is None:
            track_patterns = self.enable_learning
            
        print(f"📝 TextRetriever - wyszukiwanie: '{query}' (top_k={top_k}, threshold={score_threshold})")
        
        # Generuj embedding dla zapytania tekstowego
        query_embedding = self.embedder.get_text_embedding(query)
        if query_embedding is None:
            print("❌ Nie udało się wygenerować embeddingu dla zapytania tekstowego")
            return []
        
        # Wyszukiwanie tekstowe
        results = self._search_text_nodes(
            query_embedding, 
            top_k=top_k,
            score_threshold=score_threshold,
            source_filter=source_filter,
            subject_filter=subject_filter
        )
        
        # WYKORZYSTANIE FUNKCJONALNOŚCI LearningPatternTracker (tylko jeśli włączone)
        if track_patterns and self.learning_tracker and results:
            # 1. Zbierz ID węzłów z wyników
            retrieved_node_ids = [result['data']['id'] for result in results]
            
            # 2. Zapisz sesję zapytania z feedbackiem (główna funkcjonalność)
            self.learning_tracker.record_query_session(
                query=query,
                retrieved_nodes=retrieved_node_ids,
                user_feedback=user_feedback
            )
        
        print(f"✅ Znaleziono {len(results)} wyników tekstowych")
        return results
    
    def provide_feedback(self, query: str, results: List[Dict[str, Any]], 
                        useful_nodes: List[str] = None, 
                        not_useful_nodes: List[str] = None,
                        additional_feedback: Dict[str, Any] = None):
        """
        Umożliwia dostarczenie feedbacku po wyszukiwaniu dla poprawy uczenia się
        (Dostępne tylko gdy enable_learning=True)
        """
        if not self.learning_tracker:
            print("⚠️ Feedback wymaga włączenia learning_tracker (enable_learning=True)")
            return
            
        print(f"📝 Otrzymuję feedback dla zapytania: '{query}'")
        
        # Przygotuj feedback
        feedback = {}
        if useful_nodes:
            feedback['useful_nodes'] = useful_nodes
            print(f"  ✅ Przydatne węzły: {len(useful_nodes)}")
        
        if not_useful_nodes:
            feedback['not_useful_nodes'] = not_useful_nodes
            print(f"  ❌ Nieprzydatne węzły: {len(not_useful_nodes)}")
        
        if additional_feedback:
            feedback.update(additional_feedback)
        
        # Zbierz ID wszystkich węzłów z wyników
        retrieved_node_ids = [result['data']['id'] for result in results]
        
        # Zapisz sesję z feedbackiem - automatycznie uruchomi uczenie się
        self.learning_tracker.record_query_session(
            query=query,
            retrieved_nodes=retrieved_node_ids,
            user_feedback=feedback
        )
        
        print("✅ Feedback zapisany i przetworzony przez system uczenia się")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Zwraca szczegółowe statystyki uczenia się i wzorców użycia
        (Dostępne tylko gdy enable_learning=True)
        """
        if not self.learning_tracker:
            return {'error': 'Learning tracker nie jest włączony'}
            
        print("📊 Zbieranie statystyk uczenia się...")
        
        # Analiza wzorców użycia z LearningPatternTracker
        usage_stats = self.learning_tracker.analyze_usage_patterns()
        
        # Odkryj aktualne klastry
        current_clusters = self.learning_tracker.discover_semantic_clusters()
        
        # Przygotuj szczegółowe statystyki
        stats = {
            'query_statistics': {
                'total_queries': len(self.learning_tracker.query_history),
                'feedback_sessions': len([s for s in self.learning_tracker.query_history if s['feedback']]),
                'recent_queries': [s['query'] for s in self.learning_tracker.query_history[-5:]]
            },
            'pattern_learning': {
                'co_occurrence_patterns': len(self.learning_tracker.co_occurrence_patterns),
                'temporal_patterns': len(self.learning_tracker.temporal_patterns),
                'strongest_co_occurrences': sorted(
                    [(str(k), v) for k, v in self.learning_tracker.co_occurrence_patterns.items()], 
                    key=lambda x: x[1], reverse=True
                )[:10]
            },
            'semantic_clusters': {
                'cluster_count': len(current_clusters),
                'cluster_details': {
                    str(cluster_id): {
                        'node_count': len(nodes),
                        'sample_nodes': list(nodes)[:5]
                    }
                    for cluster_id, nodes in current_clusters.items()
                }
            },
            'popular_nodes': usage_stats.get('popular_nodes', []),
            'auto_learning_stats': usage_stats.get('auto_learning_stats', {})
        }
        
        return stats
    
    def optimize_graph(self) -> List[str]:
        """
        Ręczne uruchomienie optymalizacji grafu
        (Dostępne tylko gdy enable_learning=True)
        """
        if not self.learning_tracker:
            print("⚠️ Optymalizacja grafu wymaga włączenia learning_tracker")
            return []
            
        print("🔧 Uruchamiam manualną optymalizację grafu...")
        optimizations = self.learning_tracker.auto_optimize_graph()
        
        for optimization in optimizations:
            print(f"  ✨ {optimization}")
        
        return optimizations
    
    def discover_clusters(self) -> Dict[int, set]:
        """
        Ręczne odkrywanie klastrów semantycznych
        (Dostępne tylko gdy enable_learning=True)
        """
        if not self.learning_tracker:
            print("⚠️ Odkrywanie klastrów wymaga włączenia learning_tracker")
            return {}
            
        print("🔍 Odkrywam klastry semantyczne...")
        clusters = self.learning_tracker.discover_semantic_clusters()
        
        print(f"📊 Znaleziono {len(clusters)} klastrów:")
        for cluster_id, nodes in clusters.items():
            print(f"  Klaster {cluster_id}: {len(nodes)} węzłów")
        
        return clusters
    
    def enable_advanced_learning(self):
        """
        Włącza zaawansowane funkcje uczenia się w czasie działania
        """
        if not self.learning_tracker:
            self.learning_tracker = LearningPatternTracker(self.connector)
            self.enable_learning = True
            print("✅ Zaawansowane uczenie się zostało włączone")
        else:
            print("ℹ️ Zaawansowane uczenie się już jest włączone")
    
    def disable_advanced_learning(self):
        """
        Wyłącza zaawansowane funkcje uczenia się w czasie działania
        """
        if self.learning_tracker:
            self.learning_tracker.close()
            self.learning_tracker = None
            self.enable_learning = False
            print("✅ Zaawansowane uczenie się zostało wyłączone")
        else:
            print("ℹ️ Zaawansowane uczenie się już jest wyłączone")
    
    def run_advanced_analytics(self):
        """
        Uruchamia wszystkie zaawansowane funkcje analityczne z AdvancedRAGSystem
        (Dostępne tylko gdy enable_learning=True)
        """
        if not self.learning_tracker:
            print("⚠️ Zaawansowana analityka wymaga włączenia learning_tracker")
            return {}
        
        print("🚀 Uruchamiam zaawansowaną analitykę...")
        
        # Uruchom automatyczną optymalizację grafu
        print("🔧 Optymalizacja grafu...")
        optimizations = self.learning_tracker.auto_optimize_graph()
        for optimization in optimizations:
            print(f"  ✨ {optimization}")
        
        # Odkryj klastry semantyczne
        print("🔍 Odkrywanie klastrów semantycznych...")
        clusters = self.learning_tracker.discover_semantic_clusters()
        print(f"  📊 Znaleziono {len(clusters)} klastrów semantycznych")
        for cluster_id, nodes in list(clusters.items())[:3]:
            print(f"    Klaster {cluster_id}: {len(nodes)} węzłów")
        
        # Analizuj wzorce użycia
        print("📈 Analiza wzorców użycia...")
        usage_patterns = self.learning_tracker.analyze_usage_patterns()
        print(f"  📋 Łącznie zapytań: {usage_patterns['total_queries']}")
        print(f"  💬 Sesji z feedbackiem: {usage_patterns['feedback_sessions']}")
        print(f"  🔗 Wzorców współwystępowania: {usage_patterns['auto_learning_stats']['co_occurrence_patterns']}")
        
        # Pokaż najpopularniejsze węzły
        if usage_patterns['popular_nodes']:
            print("  🏆 Najpopularniejsze węzły:")
            for node in usage_patterns['popular_nodes'][:3]:
                print(f"    - {node['id']}: {node['usage']} użyć")
        
        return {
            'optimizations': optimizations,
            'clusters': clusters,
            'usage_patterns': usage_patterns
        }
    
    def _search_text_nodes(self, query_embedding: List[float], 
                          top_k: int = 3, 
                          score_threshold: float = 0.7,
                          source_filter: List[str] = None,
                          subject_filter: List[str] = None) -> List[Dict[str, Any]]:
        """
        Wyszukiwanie w węzłach TextNode używając cosine similarity z uwzględnieniem relacji SIMILAR_TO
        """
        # Zwiększamy top_n żeby mieć więcej kandydatów do analizy relacji
        top_n = max(top_k * 3, 30)
        
        # Buduj klauzulę WHERE z filtrami
        where_conditions = ["n:TextNode", "n.embedding IS NOT NULL"]
        
        if source_filter:
            source_filter_str = "', '".join(source_filter)
            where_conditions.append(f"n.source IN ['{source_filter_str}']")
        
        if subject_filter:
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
        ) AS cosine_score
        WHERE cosine_score > $threshold
        RETURN n.id as id, n.text as text, n.path as path, n.source as source, 
               n.type as data_type, n.base64 as base64_data, cosine_score
        ORDER BY cosine_score DESC
        LIMIT $top_n
        '''
        
        with self.driver.session() as session:
            result = session.run(search_query, 
                               query_emb=query_embedding.tolist(), 
                               threshold=score_threshold, 
                               top_n=top_n)
            
            # Przygotuj listę kandydatów z hybrydowym score
            candidates = []
            for record in result:
                source_type = record['source'] or 'unknown'
                cosine_score = record['cosine_score']
                
                hybrid_score = self.hybrid_scorer.calculate_weight(
                    embedding_similarity=cosine_score,
                    source_type=source_type
                )
                
                candidates.append({
                    'data': {
                        'id': record['id'],
                        'text': record['text'],
                        'path': record['path'],
                        'source': record['source'] or 'unknown',
                        'type': record['data_type'],
                        'base64': record['base64_data']
                    },
                    'score': hybrid_score,
                    'cosine_score': cosine_score,
                    'source_type': source_type,
                    'retrieval_method': 'hybrid_similarity',
                    'node_type': 'TextNode',
                    'boosted': False
                })
            
            # Sortuj kandydatów według score
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Algorytm wyboru z uwzględnieniem relacji SIMILAR_TO
            final_results = []
            
            while len(final_results) < top_k and candidates:
                # Znajdź kandydata z najwyższym score
                best_candidate = candidates.pop(0)
                final_results.append(best_candidate)
                
                # Sprawdź relacje SIMILAR_TO dla nowo dodanego chunka
                best_id = best_candidate['data']['id']
                similar_ids = self._get_similar_chunks(session, best_id)
                
                # Podbij score dla chunks w relacji SIMILAR_TO (tylko jeśli nie były jeszcze boostowane)
                for candidate in candidates:
                    if (candidate['data']['id'] in similar_ids and not candidate['boosted']):
                        candidate['score'] *= 1.1
                        candidate['boosted'] = True
                
                # Posortuj ponownie kandydatów po potencjalnym boostowaniu
                candidates.sort(key=lambda x: x['score'], reverse=True)
            
            return final_results
    
    def _get_similar_chunks(self, session, chunk_id: str) -> List[str]:
        """
        Pobiera ID chunków połączonych relacją SIMILAR_TO z danym chunkiem
        """
        similar_query = '''
        MATCH (n {id: $chunk_id})-[:SIMILAR_TO]-(similar)
        RETURN similar.id as similar_id
        '''
        
        result = session.run(similar_query, chunk_id=chunk_id)
        similar_ids = [record['similar_id'] for record in result]
        
        return similar_ids
    
    def close(self):
        """
        Zamyka połączenia i czyści zasoby
        """
        if hasattr(self, 'learning_tracker') and self.learning_tracker:
            self.learning_tracker.close()
        print("✅ TextRetriever zamknięty")