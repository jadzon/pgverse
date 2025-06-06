import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from neo4j import GraphDatabase
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy import sympify
import cohere
import json
from typing import List, Dict, Optional, Tuple
import logging

NEO4J_URI = "neo4j+s://335a260d.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "4RMc8un8Yjzx9oy3_l2fDw5pNbuKuNdGjLHFI4a_EEU"
COHERE_API_KEY = "xSywHzTHlEcq51tOI8rpxwWwtDdQnio5H7pPnuxs"  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LaTeXToGraph:
    """Parser LaTeX do grafu matematycznego"""
    
    def __init__(self):
        self.node_counter = 0
        self.graph = nx.DiGraph()
        
    def parse_latex_to_ast(self, latex_formula: str):
        """Konwertuje LaTeX do AST używając SymPy"""
        try:
            latex_formula = latex_formula.strip()
            if not latex_formula:
                raise ValueError("Pusty wzór LaTeX")
                
            expr = parse_latex(latex_formula)
            return expr
        except Exception as e:
            logger.warning(f"Błąd parsowania LaTeX '{latex_formula}': {e}, próba fallback")
            try:
                cleaned = latex_formula.replace('\\', '').replace('{', '').replace('}', '')
                cleaned = cleaned.replace('frac', '/').replace('sqrt', 'sqrt')
                return sympify(cleaned)
            except Exception as e2:
                logger.error(f"Fallback również nieudany: {e2}")
                return sp.Symbol('x')
    
    def build_graph_from_ast(self, expr):
        """Buduje graf z AST wyrażenia"""
        self.graph.clear()
        self.node_counter = 0
        return self._process_node(expr)
    
    def _process_node(self, node):
        """Rekurencyjnie przetwarza węzły AST"""
        node_id = self.node_counter
        self.node_counter += 1
        
        node_type = type(node).__name__
        node_attrs = {
            'id': node_id,
            'type': node_type,
            'value': str(node),
            'is_leaf': len(node.args) == 0 if hasattr(node, 'args') else True,
            'latex_repr': str(node),
            'complexity': self._calculate_complexity(node)
        }
        
        self.graph.add_node(node_id, **node_attrs)
        
        if hasattr(node, 'args') and node.args:
            for i, child in enumerate(node.args):
                child_id = self._process_node(child)
                self.graph.add_edge(node_id, child_id, 
                                  relation=f'arg_{i}',
                                  weight=1.0 / (i + 1))
        
        return node_id
    
    def _calculate_complexity(self, node):
        """Oblicza złożoność węzła"""
        if not hasattr(node, 'args') or not node.args:
            return 1
        return 1 + sum(self._calculate_complexity(child) for child in node.args)

class MathFormulaGNN(nn.Module):
    """Graph Neural Network dla wzorów matematycznych"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding(x)
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.layer_norm(x)
        
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding

class CohereMultilingualEmbedder:
    """Wrapper dla Cohere Multilingual API z obsługą błędów"""
    
    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)
        self.model = "embed-multilingual-v3.0"
        
    def get_text_embedding(self, text: str, input_type: str = "search_document") -> np.ndarray:
        """Pobiera embedding tekstowy z Cohere"""
        try:
            if not text or not text.strip():
                logger.warning("Pusty tekst, zwracam zerowy embedding")
                return np.zeros(1024)
                
            response = self.client.embed(
                texts=[text.strip()],
                model=self.model,
                input_type=input_type,
                embedding_types=["float"]
            )
            
            if hasattr(response, 'embeddings') and response.embeddings:
                return np.array(response.embeddings[0])
            else:
                logger.error("Nieprawidłowa odpowiedź z Cohere API")
                return np.zeros(1024)
                
        except Exception as e:
            logger.error(f"Błąd podczas pobierania embeddingu z Cohere: {e}")
            return np.zeros(1024)

class MathGraphEmbeddingSystem:
    """Główny system do tworzenia embeddingów matematycznych"""
    
    def __init__(self, cohere_api_key: str, vocab_size: int = 200):
        self.latex_parser = LaTeXToGraph()
        self.cohere_embedder = CohereMultilingualEmbedder(cohere_api_key)
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_size = vocab_size
        self.gnn_model = MathFormulaGNN(vocab_size)
        
        # Warstwa fuzji embeddingów
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        
    def graph_to_torch_data(self, graph: nx.DiGraph) -> Data:
        """Konwertuje NetworkX graf do PyTorch Geometric Data"""
        if not graph.nodes():
            return Data(x=torch.tensor([[1]], dtype=torch.long), 
                       edge_index=torch.empty((2, 0), dtype=torch.long))
            
        x = torch.tensor([self.vocab.get(attrs['type'], 1)
                         for _, attrs in graph.nodes(data=True)], dtype=torch.long)
        
        if graph.edges():
            edge_index = torch.tensor(list(graph.edges())).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        return Data(x=x, edge_index=edge_index)
    
    def create_math_embedding(self, latex_formula: str, context_text: str = "") -> np.ndarray:
        """Tworzy embedding łączący strukturę matematyczną i kontekst tekstowy"""
        
        # 1. Embedding strukturalny z GNN
        try:
            expr = self.latex_parser.parse_latex_to_ast(latex_formula)
            root_id = self.latex_parser.build_graph_from_ast(expr)
            graph = self.latex_parser.graph
            
            torch_data = self.graph_to_torch_data(graph)
            
            with torch.no_grad():
                if torch_data.x.size(0) > 0:
                    batch = torch.zeros(torch_data.x.size(0), dtype=torch.long)
                    structural_emb = self.gnn_model(torch_data.x, torch_data.edge_index, batch)
                    structural_emb = structural_emb.squeeze().numpy()
                else:
                    structural_emb = np.zeros(128)
        except Exception as e:
            logger.error(f"Błąd tworzenia embeddingu strukturalnego: {e}")
            structural_emb = np.zeros(128)
        
        # 2. Embedding tekstowy z Cohere
        if context_text:
            text_emb = self.cohere_embedder.get_text_embedding(context_text)
        else:
            text_emb = np.zeros(1024)
        
        # 3. Fuzja embeddingów
        try:
            combined = np.concatenate([text_emb, structural_emb])
            combined_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                fused_embedding = self.fusion_layer(combined_tensor)
                
            return fused_embedding.squeeze().numpy()
        except Exception as e:
            logger.error(f"Błąd fuzji embeddingów: {e}")
            combined = np.concatenate([text_emb[:64], structural_emb[:64]])
            return combined / (np.linalg.norm(combined) + 1e-8)

class Neo4jMathStore:
    """Klasa do zarządzania bazą Neo4j z wzorami matematycznymi"""
    
    def __init__(self, uri: str, user: str, password: str, cohere_api_key: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_system = MathGraphEmbeddingSystem(cohere_api_key)
        self._initialize_database()
        
    def _initialize_database(self):
        """Inicjalizuje ograniczenia i indeksy w Neo4j"""
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT formula_id IF NOT EXISTS FOR (f:Formula) REQUIRE f.id IS UNIQUE")
                
                session.run("""
                    CREATE VECTOR INDEX formula_embeddings IF NOT EXISTS
                    FOR (f:Formula) ON (f.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 128,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                
                logger.info("Baza danych zainicjalizowana pomyślnie")
                
            except Exception as e:
                logger.warning(f"Niektóre indeksy mogą już istnieć: {e}")
    
    def store_math_formula(self, formula_id: str, latex_formula: str, 
                          context_text: str = "", metadata: Dict = None) -> bool:
        """Zapisuje wzór matematyczny z embeddingiem do Neo4j"""
        
        try:
            if not formula_id or not latex_formula:
                logger.error("ID wzoru i wzór LaTeX są wymagane")
                return False
            
            embedding = self.embedding_system.create_math_embedding(latex_formula, context_text)
            
            with self.driver.session() as session:
                session.run("""
                    MERGE (f:Formula {id: $formula_id})
                    SET f.latex = $latex,
                        f.context = $context,
                        f.embedding = $embedding,
                        f.timestamp = timestamp(),
                        f.metadata = $metadata,
                        f.category = $category
                """, 
                formula_id=formula_id,
                latex=latex_formula,
                context=context_text,
                embedding=embedding.tolist(),
                metadata=json.dumps(metadata or {}),
                category=metadata.get('category', 'unknown') if metadata else 'unknown'
                )
                
            logger.info(f"✓ Zapisano wzór: {formula_id}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Błąd zapisywania wzoru {formula_id}: {e}")
            return False
    
    def find_similar_formulas(self, query_latex: str, query_context: str = "", 
                            top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """Znajduje podobne wzory używając similarity search"""
        
        try:
            query_embedding = self.embedding_system.create_math_embedding(query_latex, query_context)
            
            with self.driver.session() as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes('formula_embeddings', $top_k, $query_emb)
                    YIELD node, score
                    WHERE score > $threshold
                    RETURN node.id as formula_id, 
                           node.latex as latex, 
                           node.context as context,
                           node.metadata as metadata,
                           node.category as category,
                           score
                    ORDER BY score DESC
                """, 
                query_emb=query_embedding.tolist(),
                threshold=threshold,
                top_k=top_k
                )
                
                results = []
                for record in result:
                    results.append({
                        'formula_id': record['formula_id'],
                        'latex': record['latex'],
                        'context': record['context'],
                        'metadata': json.loads(record['metadata']) if record['metadata'] else {},
                        'category': record['category'],
                        'similarity_score': record['score']
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Błąd wyszukiwania podobnych wzorów: {e}")
            return []
    
    def close(self):
        """Zamyka połączenie z bazą danych"""
        self.driver.close()

def main():
    """Uproszczona funkcja główna"""
    
    # Sprawdź czy dane konfiguracyjne zostały uzupełnione
    if NEO4J_PASSWORD == "your_password_here":
        print("❌ BŁĄD: Musisz wpisać hasło Neo4j w zmiennej NEO4J_PASSWORD")
        return
    
    if COHERE_API_KEY == "your_cohere_api_key_here":
        print("❌ BŁĄD: Musisz wpisać klucz API Cohere w zmiennej COHERE_API_KEY")
        return
    
    # Inicjalizacja systemu
    math_store = Neo4jMathStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, COHERE_API_KEY)
    
    try:
        print("🧮 System wzorów matematycznych uruchomiony!")
        print("Przykłady użycia:")
        
        # Dodaj przykładowy wzór
        success = math_store.store_math_formula(
            "test_formula",
            r"x^2 + y^2 = z^2",
            "Twierdzenie Pitagorasa",
            {"category": "geometry"}
        )
        
        if success:
            print("✓ Dodano przykładowy wzór")
            
            results = math_store.find_similar_formulas(r"a^2 + b^2 = c^2", "geometria")
            print(f"✓ Znaleziono {len(results)} podobnych wzorów")
            
            for result in results:
                print(f"  - {result['formula_id']}: {result['latex']} (podobieństwo: {result['similarity_score']:.3f})")
        
        while True:
            print("\n" + "="*50)
            choice = input("1. Dodaj wzór  2. Wyszukaj  3. Wyjście\nWybór: ").strip()
            
            if choice == '1':
                formula_id = input("ID wzoru: ").strip()
                latex = input("LaTeX: ").strip()
                context = input("Kontekst: ").strip()
                
                if formula_id and latex:
                    success = math_store.store_math_formula(formula_id, latex, context)
                    print("✓ Zapisano" if success else "✗ Błąd")
                else:
                    print("❌ ID i LaTeX są wymagane")
                    
            elif choice == '2':
                query_latex = input("Wzór do wyszukania: ").strip()
                if query_latex:
                    results = math_store.find_similar_formulas(query_latex)
                    if results:
                        for result in results:
                            print(f"- {result['latex']} (podobieństwo: {result['similarity_score']:.3f})")
                    else:
                        print("Nie znaleziono podobnych wzorów")
                        
            elif choice == '3':
                break
    
    except Exception as e:
        logger.error(f"Błąd: {e}")
    finally:
        math_store.close()
        print("Połączenie zamknięte")

if __name__ == "__main__":
    main()
