from neo4j import GraphDatabase
import time
from collections import defaultdict
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy import sympify
import cohere
import json
from typing import List, Dict, Optional, Tuple
import logging
import hashlib
import re

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    """Wrapper dla Cohere Multilingual API v4 z obsługą błędów"""
    
    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)
        # AKTUALIZACJA: Cohere v4
        self.model = "embed-multilingual-v4.0"  # Zmienione z v3.0 na v4.0
        
    def get_text_embedding(self, text: str, input_type: str = "search_document") -> np.ndarray:
        """Pobiera embedding tekstowy z Cohere v4"""
        try:
            if not text or not text.strip():
                logger.warning("Pusty tekst, zwracam zerowy embedding")
                return np.zeros(1024)  # v4 może mieć inny wymiar
                
            response = self.client.embed(
                texts=[text.strip()],
                model=self.model,
                input_type=input_type,
                embedding_types=["float"]
            )
            
            if hasattr(response, 'embeddings') and response.embeddings:
                embedding = np.array(response.embeddings[0])
                # NOWE: Sprawdź wymiar embeddingu v4
                logger.info(f"Cohere v4 embedding wymiar: {embedding.shape}")
                return embedding
            else:
                logger.error("Nieprawidłowa odpowiedź z Cohere API v4")
                return np.zeros(1024)
                
        except Exception as e:
            logger.error(f"Błąd podczas pobierania embeddingu z Cohere v4: {e}")
            return np.zeros(1024)

class JSONMathProcessor:
    """Przetwarza pliki JSON zawierające wzory matematyczne"""
    
    def __init__(self, cohere_api_key: str):
        self.cohere_embedder = CohereMultilingualEmbedder(cohere_api_key)
        self.latex_parser = LaTeXToGraph()
    
    def detect_json_formulas(self, json_data):
        """Wykrywa wzory matematyczne w strukturach JSON"""
        formulas = []
        
        def extract_formulas_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    # Sprawdź czy klucz sugeruje wzór matematyczny
                    if any(keyword in key.lower() for keyword in 
                          ['formula', 'equation', 'math', 'latex', 'wzor', 'rownanie']):
                        if isinstance(value, str) and self.is_mathematical_expression(value):
                            formulas.append({
                                'path': new_path,
                                'formula': value,
                                'context': f"JSON field: {key}"
                            })
                    
                    extract_formulas_recursive(value, new_path)
                    
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_formulas_recursive(item, f"{path}[{i}]")
                    
            elif isinstance(obj, str):
                # Sprawdź czy string zawiera wzór LaTeX
                if self.is_mathematical_expression(obj):
                    formulas.append({
                        'path': path,
                        'formula': obj,
                        'context': f"JSON string at {path}"
                    })
        
        extract_formulas_recursive(json_data)
        return formulas

    def is_mathematical_expression(self, text):
        """Sprawdza czy tekst zawiera wyrażenie matematyczne"""
        math_indicators = [
            r'\\frac', r'\\sqrt', r'\\sum', r'\\int', r'\\alpha', r'\\beta',
            r'\$.*\$', r'\\begin{equation}', r'\\begin{align}',
            r'x\^', r'y\^', r'z\^', r'=', r'\+', r'\-', r'\*', r'/'
        ]
        
        for pattern in math_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def process_json_file(self, json_file_path: str) -> List[Dict]:
        """Przetwarza plik JSON i wyciąga wzory matematyczne"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Wykryj wzory w JSON
            formulas = self.detect_json_formulas(json_data)
            
            processed_formulas = []
            for formula_info in formulas:
                try:
                    # Utwórz embedding dla wzoru
                    embedding = self.cohere_embedder.get_text_embedding(
                        f"{formula_info['formula']} {formula_info['context']}"
                    )
                    
                    processed_formulas.append({
                        'id': hashlib.md5(f"{formula_info['path']}_{formula_info['formula']}".encode()).hexdigest(),
                        'latex': formula_info['formula'],
                        'context': formula_info['context'],
                        'json_path': formula_info['path'],
                        'embedding': embedding.tolist(),
                        'source_file': json_file_path
                    })
                    
                except Exception as e:
                    logger.error(f"Błąd przetwarzania wzoru {formula_info['formula']}: {e}")
            
            return processed_formulas
            
        except Exception as e:
            logger.error(f"Błąd przetwarzania pliku JSON {json_file_path}: {e}")
            return []

class MathGraphEmbeddingSystem:
    """Główny system do tworzenia embeddingów matematycznych"""
    
    def __init__(self, cohere_api_key: str, vocab_size: int = 200):
        self.latex_parser = LaTeXToGraph()
        self.cohere_embedder = CohereMultilingualEmbedder(cohere_api_key)
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_size = vocab_size
        self.gnn_model = MathFormulaGNN(vocab_size)
        
        # Warstwa fuzji embeddingów - dostosowana do v4
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024 + 128, 512),  # v4 może mieć inny wymiar
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
        
        # 2. Embedding tekstowy z Cohere v4
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
        self.json_processor = JSONMathProcessor(cohere_api_key)
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
    
    def process_and_store_json_formulas(self, json_file_path: str) -> int:
        """Przetwarza plik JSON i zapisuje znalezione wzory do Neo4j"""
        formulas = self.json_processor.process_json_file(json_file_path)
        stored_count = 0
        
        for formula in formulas:
            if self.store_math_formula(
                formula['id'],
                formula['latex'],
                formula['context'],
                {
                    'source_file': formula['source_file'],
                    'json_path': formula['json_path'],
                    'category': 'json_formula'
                }
            ):
                stored_count += 1
        
        return stored_count
    
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
    Enhanced with mathematical formula support.
    """
    def __init__(self, connector: Neo4jConnector, similarity_threshold: float = 0.8, cohere_api_key: str = None):
        self.driver = connector.get_driver()
        self.threshold = similarity_threshold
        self.usage_patterns = defaultdict(int)
        self.last_threshold_adjustment = time.time()
        
        # Dodaj obsługę wzorów matematycznych jeśli podano klucz API
        if cohere_api_key:
            self.math_store = Neo4jMathStore(
                connector.driver.uri, 
                connector.driver.auth[0], 
                connector.driver.auth[1], 
                cohere_api_key
            )
            self.latex_parser = LaTeXToGraph()
        else:
            self.math_store = None
            self.latex_parser = None

    def close(self):
        if self.math_store:
            self.math_store.close()
        self.driver.close()

    def insert_node(self, node_id: str, data_type: str, text: str, embedding: list = None, 
                   path: str = None, source: str = "unknown", base64_data: str = None):
        """
        Inserts or updates a Chunk node with given properties, including optional embedding, 
        source type, and base64 data for images/formulas/tables.
        Enhanced with mathematical formula support.
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

    def insert_math_formula(self, formula_id: str, latex_formula: str, context: str = "", metadata: Dict = None) -> bool:
        """Nowa metoda dla wzorów matematycznych"""
        if not self.math_store:
            logger.error("Math store nie jest zainicjalizowany - brak klucza API Cohere")
            return False
        
        return self.math_store.store_math_formula(formula_id, latex_formula, context, metadata)

    def process_json_formulas(self, json_file_path: str) -> int:
        """Przetwarza wzory z pliku JSON"""
        if not self.math_store:
            logger.error("Math store nie jest zainicjalizowany - brak klucza API Cohere")
            return 0
        
        return self.math_store.process_and_store_json_formulas(json_file_path)

    def find_similar_math_formulas(self, query_latex: str, query_context: str = "", 
                                  top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """Nowa metoda wyszukiwania podobnych wzorów"""
        if not self.math_store:
            logger.error("Math store nie jest zainicjalizowany - brak klucza API Cohere")
            return []
        
        return self.math_store.find_similar_formulas(query_latex, query_context, top_k, threshold)

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
        TextNode -> TextNode, ImageNode -> ImageNode, etc.
        """
        # Separate queries for each node type
        node_types = ['TextNode', 'ImageNode', 'FormulaNode', 'TableNode']
        
        for node_type in node_types:
            query = f'''
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
            
            with self.driver.session() as session:
                result = session.run(query, threshold=self.threshold)
                summary = result.consume()
                print(f"Utworzono/zaktualizowano {summary.counters.relationships_created} relacji podobieństwa dla {node_type}")

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
        
        # Statystyki wzorów matematycznych
        formula_stats_query = '''
        MATCH (f:Formula)
        RETURN count(f) as total_formulas,
               avg(f.usage_count) as avg_formula_usage,
               collect(DISTINCT f.category)[0..5] as formula_categories
        '''
        
        with self.driver.session() as session:
            node_stats = [dict(record) for record in session.run(node_stats_query)]
            relation_stats = [dict(record) for record in session.run(relation_stats_query)]
            source_stats = [dict(record) for record in session.run(source_stats_query)]
            
            # Statystyki wzorów
            formula_result = session.run(formula_stats_query)
            formula_stats = dict(formula_result.single()) if formula_result.peek() else {}
            
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
                'formula_statistics': formula_stats,
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
        """
        print("Rozpoczynam konserwację grafu...")
        
        # Relacje podobieństwa w ramach tego samego typu węzłów
        self.create_relations()
        
        # Relacje cross-modalne
        self.create_multimodal_relations()
        
        # Zastosuj zanikanie relacji
        self.decay_relationships()
        
        # Dostosuj próg adaptacyjnie
        self.adaptive_threshold_adjustment()
        
        # Wyczyść stare relacje
        self.prune_old()
        
        print("Konserwacja grafu zakończona.")

class HybridTextRetriever:
    """
    Retriever tekstowy używający podobieństwa kosinusowego z grafem Neo4j.
    Obsługuje zarówno węzły Chunk jak i TextChunk dla kompatybilności.
    Enhanced with learning capabilities and mathematical formula support.
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

    def search_math_formulas(self, query_latex: str, query_context: str = "", 
                           top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """
        Nowa metoda wyszukiwania wzorów matematycznych.
        """
        if self.graph_builder and self.graph_builder.math_store:
            return self.graph_builder.find_similar_math_formulas(
                query_latex, query_context, top_k, threshold
            )
        else:
            logger.warning("Math store nie jest dostępny w graph_builder")
            return []

class GraphPruner:
    """
    Klasa do konserwacji grafu - usuwa stare relacje.
    Enhanced with learning-aware maintenance and mathematical formula support.
    """
    def __init__(self, connector: Neo4jConnector, similarity_threshold: float = 0.8, cohere_api_key: str = None):
        self.connector = connector
        self.text_graph = TextGraphBuilder(connector, similarity_threshold, cohere_api_key)
    
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
    Klasa do śledzenia wzorców uczenia się i dostosowywania grafu.
    Enhanced with mathematical formula tracking.
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
        Enhanced with mathematical formula statistics.
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
        
        # Statystyki wzorów matematycznych
        formula_stats_query = '''
        MATCH (f:Formula)
        RETURN count(f) as total_formulas,
               avg(f.usage_count) as avg_formula_usage,
               collect(DISTINCT f.category)[0..5] as formula_categories
        '''
        
        with self.driver.session() as session:
            result = session.run(query)
            popular_nodes = [dict(record) for record in result]
            
            formula_result = session.run(formula_stats_query)
            formula_stats = dict(formula_result.single()) if formula_result.peek() else {}
        
        return {
            'popular_nodes': popular_nodes,
            'total_queries': len(self.query_history),
            'feedback_sessions': len([s for s in self.query_history if s['feedback']]),
            'formula_statistics': formula_stats
        }
