
from typing import Dict, List, Union, Optional

class HybridWeightFunction:
    """
    Klasa realizująca hybrydową funkcję wagową dla systemu RAG,
    łączącą podobieństwo semantyczne embeddingów z wiarygodnością źródła.
    """
    
    def __init__(
        self, 
        embedding_weight: float = 0.7, 
        source_weight: float = 0.3,
        source_trust_levels: Optional[Dict[str, float]] = None
    ):
        """
        Inicjalizacja funkcji wagowej.
        
        Args:
            embedding_weight: Współczynnik wagi dla podobieństwa embeddingów (domyślnie 0.7)
            source_weight: Współczynnik wagi dla wiarygodności źródła (domyślnie 0.3)
            source_trust_levels: Słownik mapujący typy źródeł na poziomy zaufania (0.0-1.0)
        """
        self.embedding_weight = embedding_weight
        self.source_weight = source_weight
        
        # Domyślne poziomy zaufania dla różnych typów źródeł
        self.source_trust_levels = source_trust_levels or {
            "wikipedia": 0.85,
            "książka": 0.9,
            "artykuł_naukowy": 0.95,
            "blog": 0.5,
            "forum": 0.3,
            "social_media": 0.2,
            "news": 0.6,
            "unknown": 0.4  # Domyślny poziom dla nieznanego źródła
        }
    
    def get_source_trust(self, source_type: str) -> float:
        """
        Pobiera poziom zaufania dla danego typu źródła.
        
        Args:
            source_type: Typ źródła (np. "wikipedia", "blog", itp.)
            
        Returns:
            Wartość poziomu zaufania (0.0-1.0)
        """
        # Normalizacja tekstu źródła do małych liter dla lepszego dopasowania
        source_type_lower = source_type.lower()
        
        # Sprawdzenie czy typ źródła istnieje w słowniku, jeśli nie - użyj "unknown"
        for key in self.source_trust_levels.keys():
            if key in source_type_lower:
                return self.source_trust_levels[key]
        
        return self.source_trust_levels.get("unknown", 0.4)
    
    def calculate_weight(
        self, 
        embedding_similarity: float, 
        source_type: str
    ) -> float:
        """
        Oblicza hybrydową wagę łączącą podobieństwo embeddingów i wiarygodność źródła.
        
        Args:
            embedding_similarity: Podobieństwo kosynusowe embeddingów (0.0-1.0)
            source_type: Typ źródła (np. "wikipedia", "blog", itp.)
            
        Returns:
            Wynikowa waga hybrydowa (0.0-1.0)
        """
        # Pobranie poziomu zaufania dla źródła
        source_trust = self.get_source_trust(source_type)
        
        # Obliczenie hybrydowej wagi
        hybrid_weight = (
            self.embedding_weight * embedding_similarity +
            self.source_weight * source_trust
        )
        
        return hybrid_weight
    
    def rerank_results(
        self,
        query_results: List[Dict[str, Union[str, float]]],
        default_source_type: str = "unknown"
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Przelicza wagi wyników wyszukiwania i sortuje je według nowej wagi hybrydowej.
        
        Args:
            query_results: Lista wyników z atrybutami 'id', 'text', 'score' i opcjonalnie 'source_type'
            default_source_type: Domyślny typ źródła do użycia, jeśli nie podano
            
        Returns:
            Posortowana lista wyników według hybrydowej wagi
        """
        # Obliczenie hybrydowych wag dla każdego wyniku
        for result in query_results:
            source_type = result.get("source_type", default_source_type)
            embedding_similarity = result["score"]
            
            # Obliczenie i dodanie nowej wagi hybrydowej
            result["hybrid_score"] = self.calculate_weight(
                embedding_similarity,
                source_type
            )
        
        # Sortowanie wyników według nowej wagi hybrydowej (malejąco)
        reranked_results = sorted(
            query_results,
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        
        return reranked_results