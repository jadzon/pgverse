from neo4j import GraphDatabase
from .graph import Neo4jConnector


class Neo4jCleaner:
    """
    Klasa do czyszczenia bazy danych Neo4j - usuwa relacje i węzły.
    Przeznaczona do użycia w innych programach.
    """
    
    def __init__(self, connector: Neo4jConnector):
        self.connector = connector
        self.driver = connector.get_driver()
    
    def close(self):
        """Zamyka połączenie z bazą danych."""
        if hasattr(self, 'driver'):
            self.driver.close()
    
    def delete_all_relationships(self, silent: bool = False):
        """
        Usuwa wszystkie relacje z bazy danych.
        
        Args:
            silent: Jeśli True, nie wypisuje komunikatów
            
        Returns:
            int: Liczba usuniętych relacji
        """
        query = """
        MATCH ()-[r]->()
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            deleted_count = result.single()["deleted_count"]
            if not silent:
                print(f"Usunięto {deleted_count} relacji")
            return deleted_count
    
    def delete_specific_relationships(self, relationship_types: list, silent: bool = False):
        """
        Usuwa tylko określone typy relacji.
        
        Args:
            relationship_types: Lista typów relacji do usunięcia
            silent: Jeśli True, nie wypisuje komunikatów
            
        Returns:
            int: Liczba usuniętych relacji
        """
        if not relationship_types:
            if not silent:
                print("Nie podano typów relacji do usunięcia")
            return 0
        
        rel_types_str = "|".join(relationship_types)
        query = f"""
        MATCH ()-[r:{rel_types_str}]->()
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            deleted_count = result.single()["deleted_count"]
            if not silent:
                print(f"Usunięto {deleted_count} relacji typu: {', '.join(relationship_types)}")
            return deleted_count
    
    def delete_all_chunks(self, silent: bool = False):
        """
        Usuwa wszystkie węzły typu Chunk z bazy danych.
        
        Args:
            silent: Jeśli True, nie wypisuje komunikatów
            
        Returns:
            int: Liczba usuniętych węzłów
        """
        query = """
        MATCH (n:Chunk)
        DELETE n
        RETURN count(n) as deleted_count
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            deleted_count = result.single()["deleted_count"]
            if not silent:
                print(f"Usunięto {deleted_count} węzłów typu Chunk")
            return deleted_count
    
    def delete_chunks_by_type(self, data_type: str, silent: bool = False):
        """
        Usuwa węzły Chunk określonego typu.
        
        Args:
            data_type: Typ danych do usunięcia ('text', 'image', etc.)
            silent: Jeśli True, nie wypisuje komunikatów
            
        Returns:
            int: Liczba usuniętych węzłów
        """
        query = """
        MATCH (n:Chunk {type: $data_type})
        DELETE n
        RETURN count(n) as deleted_count
        """
        
        with self.driver.session() as session:
            result = session.run(query, data_type=data_type)
            deleted_count = result.single()["deleted_count"]
            if not silent:
                print(f"Usunięto {deleted_count} węzłów typu Chunk z data_type='{data_type}'")
            return deleted_count
    
    def delete_legacy_textchunks(self, silent: bool = False):
        """
        Usuwa stare węzły typu TextChunk.
        
        Args:
            silent: Jeśli True, nie wypisuje komunikatów
            
        Returns:
            int: Liczba usuniętych węzłów
        """
        query = """
        MATCH (n:TextChunk)
        DELETE n
        RETURN count(n) as deleted_count
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            deleted_count = result.single()["deleted_count"]
            if not silent:
                print(f"Usunięto {deleted_count} starych węzłów typu TextChunk")
            return deleted_count
    
    def delete_all_nodes(self, silent: bool = False):
        """
        Usuwa wszystkie węzły z bazy danych.
        
        Args:
            silent: Jeśli True, nie wypisuje komunikatów
            
        Returns:
            dict: Słownik z liczbą usuniętych węzłów każdego typu
        """
        results = {}
        results['Chunk'] = self.delete_all_chunks(silent)
        results['TextChunk'] = self.delete_legacy_textchunks(silent)
        return results
    
    def full_cleanup(self, silent: bool = False):
        """
        Pełne czyszczenie bazy danych.
        
        Args:
            silent: Jeśli True, nie wypisuje komunikatów
            
        Returns:
            dict: Statystyki czyszczenia
        """
        if not silent:
            print("Rozpoczynanie pełnego czyszczenia bazy danych...")
        
        stats = {}
        stats['relationships_deleted'] = self.delete_all_relationships(silent)
        stats['nodes_deleted'] = self.delete_all_nodes(silent)
        
        if not silent:
            total_nodes = sum(stats['nodes_deleted'].values())
            print(f"Pełne czyszczenie zakończone. Usunięto {stats['relationships_deleted']} relacji i {total_nodes} węzłów")
        
        return stats


# Funkcje pomocnicze do użycia w innych programach

def clean_all_relationships(neo4j_uri: str, neo4j_user: str, neo4j_password: str, silent: bool = False):
    """
    Usuwa wszystkie relacje z bazy danych.
    
    Args:
        neo4j_uri: URI do bazy danych Neo4j
        neo4j_user: Nazwa użytkownika Neo4j
        neo4j_password: Hasło Neo4j
        silent: Jeśli True, nie wypisuje komunikatów
        
    Returns:
        int: Liczba usuniętych relacji
    """
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    cleaner = Neo4jCleaner(connector)
    
    try:
        return cleaner.delete_all_relationships(silent)
    finally:
        cleaner.close()
        connector.close()


def clean_all_chunks(neo4j_uri: str, neo4j_user: str, neo4j_password: str, silent: bool = False):
    """
    Usuwa wszystkie węzły Chunk z bazy danych.
    
    Args:
        neo4j_uri: URI do bazy danych Neo4j
        neo4j_user: Nazwa użytkownika Neo4j
        neo4j_password: Hasło Neo4j
        silent: Jeśli True, nie wypisuje komunikatów
        
    Returns:
        int: Liczba usuniętych węzłów
    """
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    cleaner = Neo4jCleaner(connector)
    
    try:
        return cleaner.delete_all_chunks(silent)
    finally:
        cleaner.close()
        connector.close()


def clean_chunks_by_type(neo4j_uri: str, neo4j_user: str, neo4j_password: str, data_type: str, silent: bool = False):
    """
    Usuwa węzły Chunk określonego typu.
    
    Args:
        neo4j_uri: URI do bazy danych Neo4j
        neo4j_user: Nazwa użytkownika Neo4j
        neo4j_password: Hasło Neo4j
        data_type: Typ danych do usunięcia ('text', 'image', etc.)
        silent: Jeśli True, nie wypisuje komunikatów
        
    Returns:
        int: Liczba usuniętych węzłów
    """
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    cleaner = Neo4jCleaner(connector)
    
    try:
        return cleaner.delete_chunks_by_type(data_type, silent)
    finally:
        cleaner.close()
        connector.close()


def full_database_cleanup(neo4j_uri: str, neo4j_user: str, neo4j_password: str, silent: bool = False):
    """
    Pełne czyszczenie bazy danych - usuwa wszystkie relacje i węzły.
    
    Args:
        neo4j_uri: URI do bazy danych Neo4j
        neo4j_user: Nazwa użytkownika Neo4j
        neo4j_password: Hasło Neo4j
        silent: Jeśli True, nie wypisuje komunikatów
        
    Returns:
        dict: Statystyki czyszczenia
    """
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    cleaner = Neo4jCleaner(connector)
    
    try:
        return cleaner.full_cleanup(silent)
    finally:
        cleaner.close()
        connector.close()


def clean_database(neo4j_uri: str, neo4j_user: str, neo4j_password: str, operation: str = "full_cleanup"):
    """
    Funkcja kompatybilności - czyści bazę danych.
    
    Args:
        neo4j_uri: URI do bazy danych Neo4j
        neo4j_user: Nazwa użytkownika Neo4j
        neo4j_password: Hasło Neo4j
        operation: Typ operacji (default: "full_cleanup")
    """
    connector = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    cleaner = Neo4jCleaner(connector)
    
    try:
        if operation == "delete_relationships":
            cleaner.delete_all_relationships()
        elif operation == "delete_chunks":
            cleaner.delete_all_chunks()
        elif operation == "delete_textchunks":
            cleaner.delete_legacy_textchunks()
        elif operation == "delete_all_nodes":
            cleaner.delete_all_nodes()
        elif operation == "full_cleanup":
            cleaner.full_cleanup()
        elif operation == "delete_specific_relations":
            cleaner.delete_specific_relationships(['SIMILAR_TO', 'IMAGE_SIMILAR', 'IMAGE_ILLUSTRATES', 'TEXT_ILLUSTRATED_BY'])
    finally:
        cleaner.close()
        connector.close()