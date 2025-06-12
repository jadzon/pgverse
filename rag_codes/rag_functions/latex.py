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
import os

# Konfiguracja logowania
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
            # Usuń tagi i nadmiarowe spacje
            cleaned_formula = re.sub(r'\\tag\{.*?\}', '', latex_formula).strip()
            # Zastąp \left| i \right| standardowymi znakami
            cleaned_formula = re.sub(r'\\left\|(.*?)\\right\|', r'Abs(\1)', cleaned_formula)
            
            # Jeśli \leq nadal jest wewnątrz | |, spróbuj uprościć
            if '|' in cleaned_formula and r'\leq' in cleaned_formula.split('|')[1]:
                # Prostsze czyszczenie jako fallback
                cleaned_formula = latex_formula.replace(r'\left|', '').replace(r'\right|', '')

            expr = parse_latex(cleaned_formula)
            return expr
        except Exception as e:
            logger.warning(f"Błąd parsowania LaTeX '{latex_formula}': {e}, próba fallback")
            # Logika fallback pozostaje na swoim miejscu jako ostateczność
            try:
                cleaned = latex_formula.replace('\\', '').replace('{', '').replace('}', '')
                cleaned = cleaned.replace('frac', '/').replace('leq', '<=')
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
    
    def __init__(self, vocab_size: int, embedding_dim: int = 1024, hidden_dim: int = 1024):
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

class FormulaEmbedder:
    """Klasa do tworzenia embeddingów wzorów matematycznych bez bazy danych"""
    
    def __init__(self, cohere_api_key: str):
        self.embedding_system = MathGraphEmbeddingSystem(cohere_api_key)
        self.json_processor = JSONMathProcessor(cohere_api_key)
    
    def get_formula_structural_embedding(self, latex_formula: str) -> Optional[np.ndarray]:
        """
        Tworzy embedding strukturalny dla podanego wzoru LaTeX
        
        Args:
            latex_formula: Wzór LaTeX do przetworzenia
            
        Returns:
            np.ndarray: 128-wymiarowy embedding strukturalny wzoru lub None jeśli błąd
        """
        try:
            structural_embedding = self._create_structural_only_embedding(latex_formula)
            return structural_embedding
            
        except Exception as e:
            logger.error(f"Błąd pobierania embeddingu strukturalnego dla wzoru: {e}")
            return None
    
    def _create_structural_only_embedding(self, latex_formula: str) -> np.ndarray:
        """
        Tworzy embedding tylko dla struktury wzoru (bez kontekstu tekstowego)
        
        Args:
            latex_formula: Wzór LaTeX
            
        Returns:
            np.ndarray: 128-wymiarowy embedding strukturalny
        """
        try:
            # Parsuj LaTeX do AST
            expr = self.embedding_system.latex_parser.parse_latex_to_ast(latex_formula)
            
            # Zbuduj graf z AST
            root_id = self.embedding_system.latex_parser.build_graph_from_ast(expr)
            graph = self.embedding_system.latex_parser.graph
            
            # Konwertuj graf do formatu PyTorch
            torch_data = self.embedding_system.graph_to_torch_data(graph)
            
            # Generuj embedding przez GNN
            with torch.no_grad():
                if torch_data.x.size(0) > 0:
                    batch = torch.zeros(torch_data.x.size(0), dtype=torch.long)
                    structural_emb = self.embedding_system.gnn_model(
                        torch_data.x, torch_data.edge_index, batch
                    )
                    structural_emb = structural_emb.squeeze().numpy()
                    
                    # Normalizuj embedding
                    norm = np.linalg.norm(structural_emb)
                    if norm > 0:
                        structural_emb = structural_emb / norm
                    
                    logger.info(f"Utworzono embedding strukturalny o wymiarze: {structural_emb.shape}")
                    return structural_emb
                else:
                    logger.warning("Pusty graf, zwracam zerowy embedding")
                    return np.zeros(128)
                    
        except Exception as e:
            logger.error(f"Błąd tworzenia embeddingu strukturalnego: {e}")
            return np.zeros(128)
    
    def get_formula_by_json_path(self, json_file_path: str, png_filename: str) -> Optional[np.ndarray]:
        """
        Znajduje wzór LaTeX z pliku JSON na podstawie nazwy pliku PNG i zwraca jego embedding strukturalny
        
        Args:
            json_file_path: Ścieżka do pliku JSON zawierającego wzory (może być pełna ścieżka lub nazwa)
            png_filename: Nazwa pliku PNG ze wzorem (może być pełna ścieżka lub nazwa)
            
        Returns:
            np.ndarray: 128-wymiarowy embedding strukturalny wzoru lub None jeśli nie znaleziono
        """
        try:
            # Wyciągnij nazwy plików z pełnych ścieżek
            json_filename = extract_filename_from_path(json_file_path)
            png_name = extract_filename_from_path(png_filename)
            
            logger.info(f"Szukam pliku JSON: {json_filename}")
            logger.info(f"Szukam wzoru dla: {png_name}")
            
            # Znajdź plik JSON w bieżącym katalogu i podkatalogach
            json_path = self._find_file_in_directory(json_filename)
            if not json_path:
                logger.error(f"Nie znaleziono pliku JSON: {json_filename}")
                return None
            
            # Wczytaj plik JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Sprawdź czy klucz PNG istnieje w JSON
            if png_name not in json_data:
                logger.warning(f"Nie znaleziono klucza '{png_name}' w pliku {json_path}")
                return None
            
            # Pobierz wzór LaTeX dla danego klucza PNG
            latex_formula = json_data[png_name]
            logger.info(f"Znaleziono wzór dla {png_name}: {latex_formula[:100]}...")
            
            # Oczyść wzór LaTeX z znaków formatujących
            cleaned_latex = self._clean_latex_formula(latex_formula)
            
            # Utwórz embedding strukturalny tylko dla wzoru (bez kontekstu)
            structural_embedding = self._create_structural_only_embedding(cleaned_latex)
            return structural_embedding
            
        except FileNotFoundError:
            logger.error(f"Nie znaleziono pliku JSON: {json_file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Błąd dekodowania JSON z pliku {json_file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Błąd pobierania wzoru z JSON {json_file_path}, PNG {png_filename}: {e}")
            return None
    
    def _find_file_in_directory(self, filename: str, search_path: str = ".") -> Optional[str]:
        """
        Znajduje plik w katalogu i jego podkatalogach
        
        Args:
            filename: Nazwa pliku do znalezienia
            search_path: Ścieżka do przeszukania (domyślnie bieżący katalog)
            
        Returns:
            str: Pełna ścieżka do znalezionego pliku lub None
        """
        for root, dirs, files in os.walk(search_path):
            if filename in files:
                found_path = os.path.join(root, filename)
                logger.info(f"Znaleziono plik: {found_path}")
                return found_path
        return None
    
    def _clean_latex_formula(self, latex_formula: str) -> str:
        """
        Czyści wzór LaTeX z niepotrzebnych znaków formatujących
        
        Args:
            latex_formula: Surowy wzór LaTeX z JSON
            
        Returns:
            str: Oczyszczony wzór LaTeX
        """
        try:
            # Usuń zewnętrzne znaczniki $$
            cleaned = latex_formula.strip()
            if cleaned.startswith('$$') and cleaned.endswith('$$'):
                cleaned = cleaned[2:-2].strip()
            
            # Usuń znaki nowej linii i nadmiarowe spacje
            cleaned = re.sub(r'\n+', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Usuń znaczniki \tag{...}
            cleaned = re.sub(r'\\tag\{[^}]*\}', '', cleaned)
            
            # Usuń komentarze końcowe typu (1.23)
            cleaned = re.sub(r'\s*\([0-9.]+\)\s*$', '', cleaned)
            
            logger.debug(f"Oczyszczony LaTeX: {cleaned}")
            return cleaned.strip()
            
        except Exception as e:
            logger.warning(f"Błąd podczas czyszczenia LaTeX: {e}, zwracam oryginalny")
            return latex_formula
    
    def get_multiple_formulas_from_json(self, json_file_path: str, png_filenames: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Pobiera embeddingi dla wielu wzorów z jednego pliku JSON
        
        Args:
            json_file_path: Ścieżka do pliku JSON (może być pełna ścieżka lub nazwa)
            png_filenames: Lista nazw plików PNG ze wzorami (może być pełne ścieżki lub nazwy)
            
        Returns:
            Dict[str, Optional[np.ndarray]]: Słownik {nazwa_png: embedding}
        """
        results = {}
        
        try:
            # Wyciągnij nazwy plików z pełnych ścieżek
            json_filename = extract_filename_from_path(json_file_path)
            
            # Znajdź plik JSON
            json_path = self._find_file_in_directory(json_filename)
            if not json_path:
                logger.error(f"Nie znaleziono pliku JSON: {json_filename}")
                return {extract_filename_from_path(png): None for png in png_filenames}
            
            # Wczytaj JSON raz
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            for png_filename in png_filenames:
                png_name = extract_filename_from_path(png_filename)
                if png_name in json_data:
                    latex_formula = json_data[png_name]
                    cleaned_latex = self._clean_latex_formula(latex_formula)
                    embedding = self._create_structural_only_embedding(cleaned_latex)
                    results[png_name] = embedding
                    logger.info(f"✓ Przetworzono {png_name}")
                else:
                    results[png_name] = None
                    logger.warning(f"✗ Nie znaleziono {png_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Błąd przetwarzania wielu wzorów: {e}")
            return {extract_filename_from_path(png): None for png in png_filenames}
    
    @staticmethod
    def get_formula_embedding_from_paths(json_file_path: str, png_filename: str) -> Optional[np.ndarray]:
        """
        Statyczna metoda zwracająca embedding wzoru na podstawie ścieżki JSON i nazwy PNG
        Może być używana w innych programach bez tworzenia instancji klasy
        
        Args:
            json_file_path: Ścieżka do pliku JSON (może być pełna ścieżka lub nazwa)
            png_filename: Nazwa pliku PNG (może być pełna ścieżka lub nazwa)
            
        Returns:
            np.ndarray: 512-wymiarowy embedding strukturalny wzoru lub None
        """
        try:
            # Wyciągnij nazwy plików z pełnych ścieżek
            json_filename = extract_filename_from_path(json_file_path)
            png_name = extract_filename_from_path(png_filename)
            
            logger.info(f"Szukam pliku JSON: {json_filename}")
            logger.info(f"Szukam wzoru dla: {png_name}")
            
            # Znajdź plik JSON w bieżącym katalogu i podkatalogach
            json_path = None
            for root, dirs, files in os.walk("."):
                if json_filename in files:
                    json_path = os.path.join(root, json_filename)
                    logger.info(f"Znaleziono plik JSON: {json_path}")
                    break
            
            if not json_path:
                logger.error(f"Nie znaleziono pliku JSON: {json_filename}")
                return None
            
            # Wczytaj JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        
            # Sprawdź czy klucz istnieje
            if png_name not in json_data:
                logger.warning(f"Nie znaleziono klucza '{png_name}' w {json_path}")
                return None
        
            # Pobierz wzór LaTeX
            latex_formula = json_data[png_name]
            logger.info(f"Znaleziono wzór dla {png_name}: {latex_formula[:100]}...")
        
            # Oczyść wzór z formatowania JSON
            cleaned = latex_formula.strip()
            if cleaned.startswith('$$') and cleaned.endswith('$$'):
                cleaned = cleaned[2:-2].strip()
            cleaned = re.sub(r'\n+', ' ', cleaned)
            cleaned = re.sub(r'\\tag\{[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = re.sub(r'\s*\([0-9.]+\)\s*$', '', cleaned)
        
            # Utwórz parser i model GNN
            latex_parser = LaTeXToGraph()
            gnn_model = MathFormulaGNN(200)
        
            # Parsuj LaTeX
            expr = latex_parser.parse_latex_to_ast(cleaned)
            latex_parser.build_graph_from_ast(expr)
        
            # Sprawdź czy graf nie jest pusty
            if not latex_parser.graph.nodes():
                logger.warning(f"Pusty graf dla wzoru {png_name}")
                return np.zeros(512)
        
            # Przygotuj dane dla PyTorch
            vocab = {'<PAD>': 0, '<UNK>': 1}
            x = torch.tensor([vocab.get(attrs['type'], 1) for _, attrs in latex_parser.graph.nodes(data=True)], dtype=torch.long)
        
            if latex_parser.graph.edges():
                edge_index = torch.tensor(list(latex_parser.graph.edges())).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
        
            torch_data = Data(x=x, edge_index=edge_index)
        
            # Generuj embedding
            with torch.no_grad():
                if torch_data.x.size(0) > 0:
                    batch = torch.zeros(torch_data.x.size(0), dtype=torch.long)
                    embedding = gnn_model(torch_data.x, torch_data.edge_index, batch)
                    result = embedding.squeeze().numpy()
                    
                    # Normalizuj embedding
                    norm = np.linalg.norm(result)
                    if norm > 0:
                        result = result / norm
                    
                    logger.info(f"Utworzono embedding dla {png_name}, wymiar: {result.shape}")
                    return result
                else:
                    logger.warning(f"Pusty tensor dla wzoru {png_name}")
                    return np.zeros(512)
                
        except Exception as e:
            logger.error(f"Błąd tworzenia embeddingu dla {png_filename}: {e}")
            return None
    
    @staticmethod 
    def get_latex_embedding(latex_formula: str) -> Optional[np.ndarray]:
        """
        Statyczna metoda do tworzenia embeddingu bezpośrednio z wzoru LaTeX
        Może być używana w innych programach bez konieczności pliku JSON
        
        Args:
            latex_formula: Wzór LaTeX do przetworzenia
            
        Returns:
            np.ndarray: 512-wymiarowy embedding strukturalny wzoru lub None
        """
        try:
            # Oczyść wzór LaTeX
            cleaned = latex_formula.strip()
            if cleaned.startswith('$$') and cleaned.endswith('$$'):
                cleaned = cleaned[2:-2].strip()
            cleaned = re.sub(r'\n+', ' ', cleaned)
            cleaned = re.sub(r'\\tag\{[^}]*\}', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = re.sub(r'\s*\([0-9.]+\)\s*$', '', cleaned)
            
            # Utwórz parser i model GNN
            latex_parser = LaTeXToGraph()
            gnn_model = MathFormulaGNN(200)
            
            # Parsuj LaTeX
            expr = latex_parser.parse_latex_to_ast(cleaned)
            latex_parser.build_graph_from_ast(expr)
            
            # Sprawdź czy graf nie jest pusty
            if not latex_parser.graph.nodes():
                logger.warning("Pusty graf dla wzoru LaTeX")
                return np.zeros(512)
            
            # Przygotuj dane dla PyTorch
            vocab = {'<PAD>': 0, '<UNK>': 1}
            x = torch.tensor([vocab.get(attrs['type'], 1) for _, attrs in latex_parser.graph.nodes(data=True)], dtype=torch.long)
            
            if latex_parser.graph.edges():
                edge_index = torch.tensor(list(latex_parser.graph.edges())).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            torch_data = Data(x=x, edge_index=edge_index)
            
            # Generuj embedding
            with torch.no_grad():
                if torch_data.x.size(0) > 0:
                    batch = torch.zeros(torch_data.x.size(0), dtype=torch.long)
                    embedding = gnn_model(torch_data.x, torch_data.edge_index, batch)
                    result = embedding.squeeze().numpy()
                    
                    # Normalizuj embedding
                    norm = np.linalg.norm(result)
                    if norm > 0:
                        result = result / norm
                    
                    logger.info(f"Utworzono embedding LaTeX, wymiar: {result.shape}")
                    return result
                else:
                    logger.warning("Pusty tensor dla wzoru LaTeX")
                    return np.zeros(512)
                    
        except Exception as e:
            logger.error(f"Błąd tworzenia embeddingu LaTeX: {e}")
            return None

def extract_filename_from_path(file_path: str) -> str:
    """
    Wyciąga nazwę pliku z pełnej ścieżki
    
    Args:
        file_path: Pełna ścieżka do pliku (np. "c:/folder/subfolder/file.json")
        
    Returns:
        str: Nazwa pliku (np. "file.json")
    """
    return os.path.basename(file_path)