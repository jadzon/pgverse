from neo4j import GraphDatabase

class GraphBuilder:
    """
    Builds a semantic knowledge graph in Neo4j by creating specialized relationships
    between different types of nodes (text and image) based on cosine similarity.
    """
    def __init__(self, uri, user, password, similarity_threshold=0.7):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.threshold = similarity_threshold

    def close(self):
        self.driver.close()

    def create_relationships(self):
        """
        Create type-specific relationships with weight and last_used timestamp
        between pairs of nodes based on their types (text or image).
        """
        # 1. Create IMAGE_SIMILAR relationships between images
        self._create_image_to_image_relationships()
        
        # 2. Create TEXT_ILLUSTRATES and IMAGE_DESCRIBES relationships between images and text
        self._create_image_to_text_relationships()
        
        # 3. Create standard SIMILAR_TO relationships for backwards compatibility
        self._create_generic_relationships()
    
    def _create_image_to_image_relationships(self):
        """Create relationships between similar images"""
        query = '''
        MATCH (a:Chunk), (b:Chunk)
        WHERE a.type = 'image' AND b.type = 'image'
        AND elementId(a) <> elementId(b)
        AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
        WITH a, b,
        reduce(dot = 0.0, i IN range(0, size(a.embedding)-1) |
            dot + a.embedding[i] * b.embedding[i]
        ) /
        (
            sqrt(reduce(norm_a = 0.0, i IN range(0, size(a.embedding)-1) |
            norm_a + a.embedding[i] * a.embedding[i]
            )) *
            sqrt(reduce(norm_b = 0.0, i IN range(0, size(b.embedding)-1) |
            norm_b + b.embedding[i] * b.embedding[i]
            ))
        ) AS sim
        WHERE sim >= $threshold
        MERGE (a)-[r:IMAGE_SIMILAR]->(b)
        ON CREATE SET r.weight = sim, r.last_used = timestamp()
        ON MATCH SET r.weight = sim, r.last_used = timestamp()
        '''
        with self.driver.session() as session:
            session.run(query, threshold=self.threshold)
    
    def _create_image_to_text_relationships(self):
        """Create bidirectional relationships between images and related text"""
        query = '''
        MATCH (img:Chunk), (txt:Chunk)
        WHERE img.type = 'image' AND txt.type = 'text'
        AND img.embedding IS NOT NULL AND txt.embedding IS NOT NULL
        WITH img, txt,
        reduce(dot = 0.0, i IN range(0, size(img.embedding)-1) |
            dot + img.embedding[i] * txt.embedding[i]
        ) /
        (
            sqrt(reduce(norm_img = 0.0, i IN range(0, size(img.embedding)-1) |
            norm_img + img.embedding[i] * img.embedding[i]
            )) *
            sqrt(reduce(norm_txt = 0.0, i IN range(0, size(txt.embedding)-1) |
            norm_txt + txt.embedding[i] * txt.embedding[i]
            ))
        ) AS sim
        WHERE sim >= $threshold
        // Twórz relację od obrazu do tekstu
        MERGE (img)-[r1:IMAGE_ILLUSTRATES]->(txt)
        ON CREATE SET r1.weight = sim, r1.last_used = timestamp()
        ON MATCH SET r1.weight = sim, r1.last_used = timestamp()
        
        // Twórz relację od tekstu do obrazu
        MERGE (txt)-[r2:TEXT_ILLUSTRATED_BY]->(img)
        ON CREATE SET r2.weight = sim, r2.last_used = timestamp()
        ON MATCH SET r2.weight = sim, r2.last_used = timestamp()
        '''
        with self.driver.session() as session:
            session.run(query, threshold=self.threshold)
    
    def _create_generic_relationships(self):
        """Create standard SIMILAR_TO relationships (for backwards compatibility)"""
        query = '''
        MATCH (a:Chunk), (b:Chunk)
        WHERE elementId(a) <> elementId(b)
        AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
        WITH a, b,
        reduce(dot = 0.0, i IN range(0, size(a.embedding)-1) |
            dot + a.embedding[i] * b.embedding[i]
        ) /
        (
            sqrt(reduce(norm_a = 0.0, i IN range(0, size(a.embedding)-1) |
            norm_a + a.embedding[i] * a.embedding[i]
            )) *
            sqrt(reduce(norm_b = 0.0, i IN range(0, size(b.embedding)-1) |
            norm_b + b.embedding[i] * b.embedding[i]
            ))
        ) AS sim
        WHERE sim >= $threshold
        MERGE (a)-[r:SIMILAR_TO]->(b)
        ON CREATE SET r.weight = sim, r.last_used = timestamp()
        ON MATCH SET r.weight = sim, r.last_used = timestamp()
        '''
        with self.driver.session() as session:
            session.run(query, threshold=self.threshold)

    def build_graph(self):
        """
        Convenience method to build the entire graph with all relationship types
        """
        self.create_relationships()

class HybridTextRetriever:
    """
    Performs hybrid retrieval combining vector search and graph expansion.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def retrieve(self, query_embedding, top_k=5, expand_k=3):
        """
        Returns a list of image nodes with similarity scores using specialized relationships.
        """
        # 1. Vector-based retrieval specifically for images
        vec_query = '''
        MATCH (n:Chunk)
        WHERE n.type = 'image' AND n.embedding IS NOT NULL
        WITH n,
        reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) |
            dot + n.embedding[i] * $q[i]
        ) /
        (
            sqrt(reduce(norm_n = 0.0, i IN range(0, size(n.embedding)-1) |
            norm_n + n.embedding[i] * n.embedding[i]
            )) *
            sqrt(reduce(norm_q = 0.0, i IN range(0, size($q)-1) |
            norm_q + $q[i] * $q[i]
            ))
        ) AS score
        WHERE score > 0
        RETURN elementId(n) AS node_id, n, score
        ORDER BY score DESC
        LIMIT $k
        '''

        with self.driver.session() as session:
            base = session.run(vec_query, q=query_embedding, k=top_k).data()
        
        results = []
        for record in base:
            node = record['n']
            score = record['score']
            node_id = record['node_id']
            results.append({
                'id': node_id,
                'data': node,
                'score': score
            })
            
        # 2. Graph expansion using the new specialized relationships
        graph_query = '''
        MATCH (n)-[r:IMAGE_SIMILAR|IMAGE_ILLUSTRATES|TEXT_ILLUSTRATED_BY|SIMILAR_TO]->(nbr:Chunk)
        WHERE elementId(n) = $node_id 
        AND (nbr.type = 'image' OR 
            (r.weight >= 0.75 AND nbr.type = 'text' AND EXISTS((nbr)-[:TEXT_ILLUSTRATED_BY]->(:Chunk {type: 'image'}))))
        RETURN elementId(nbr) AS nbr_id, nbr, r.weight AS score, type(r) as rel_type
        ORDER BY score DESC
        LIMIT $m
        '''
        
        # Track visited nodes to avoid re-exploring
        visited = set(item['id'] for item in results)
        
        for item in list(results):
            nbrs = self.driver.session().run(
                graph_query,
                node_id=item['id'],
                m=expand_k
            ).data()
            
            for rec in nbrs:
                nbr = rec['nbr']
                nbr_id = rec['nbr_id']
                weight = rec['score']
                rel_type = rec['rel_type']
                
                # Apply weight adjustments based on relationship type
                if nbr_id not in visited:
                    visited.add(nbr_id)
                    
                    # Add bonus for directly similar images
                    if rel_type == 'IMAGE_SIMILAR' and nbr.get('type') == 'image':
                        boost_factor = 1.2  # Boost direct image similarities
                    else:
                        boost_factor = 0.9  # Slightly discount other relationships
                    
                    results.append({
                        'id': nbr_id,
                        'data': nbr,
                        'score': weight * boost_factor,
                        'relation': rel_type
                    })

        # 3. Rerank and dedupe
        merged = {}
        for r in results:
            nid = r['id']
            if nid in merged:
                # Preferencję dajemy wynikom bezpośrednim (bez relacji)
                if 'relation' not in r:
                    merged[nid]['score'] = max(merged[nid]['score'], r['score'])
                else:
                    merged[nid]['score'] = (merged[nid]['score'] + r['score']) / 2
            else:
                merged[nid] = r
                
        # Sort by score
        final = sorted(merged.values(), key=lambda x: x['score'], reverse=True)
        return final[:top_k]

class GraphPruner:
    """
    Periodically decays weights and prunes old relationships.
    """
    def __init__(self, uri, user, password, decay_rate=1e-7, max_age_ms=30*24*3600*1000):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.decay = decay_rate
        self.max_age = max_age_ms

    def close(self):
        self.driver.close()

    def decay_weights(self):
        """
        Apply exponential decay to relationship weights for all relationship types.
        """
        query = '''
        MATCH ()-[r:SIMILAR_TO|IMAGE_SIMILAR|IMAGE_ILLUSTRATES|TEXT_ILLUSTRATED_BY]->()
        SET r.weight = r.weight * exp(-$decay * (timestamp() - r.last_used))
        '''
        with self.driver.session() as session:
            session.run(query, decay=self.decay)

    def prune_old(self):
        """
        Remove all types of relationships not used within max_age.
        """
        query = '''
        MATCH ()-[r:SIMILAR_TO|IMAGE_SIMILAR|IMAGE_ILLUSTRATES|TEXT_ILLUSTRATED_BY]->()
        WHERE r.last_used < timestamp() - $max_age
        DELETE r
        '''
        with self.driver.session() as session:
            session.run(query, max_age=self.max_age)

    def run_maintenance(self):
        self.decay_weights()
        self.prune_old()

    def get_all_relationship_types(self):
        """Return all relationship types in the database"""
        query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        with self.driver.session() as session:
            result = session.run(query)
            return [record["relationshipType"] for record in result]
        
    def decay_weights_dynamic(self):
        """Apply decay to all relationship types automatically"""
        rel_types = self.get_all_relationship_types()
        rel_pattern = "|".join(rel_types)
        
        query = f'''
        MATCH ()-[r:{rel_pattern}]->()
        WHERE exists(r.weight) AND exists(r.last_used)
        SET r.weight = r.weight * exp(-$decay * (timestamp() - r.last_used))
        '''
        with self.driver.session() as session:
            session.run(query, decay=self.decay)

class ImageRetriever:
    """
    Specialized retriever for image content, combining vector search and graph navigation.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def retrieve(self, query_embedding, top_k=5, expand_k=3):
        """
        Returns a list of image nodes with similarity scores, optimized for image retrieval.
        
        Args:
            query_embedding: The embedding vector of the query
            top_k: Number of top results to return
            expand_k: Number of related nodes to explore per result
        
        Returns:
            List of dicts with id, data (containing image path), and similarity score
        """
        # 1. Vector-based retrieval specifically for images
        vec_query = '''
            MATCH (n:Chunk)
            WHERE n.type = 'image' AND n.embedding IS NOT NULL
            WITH n,
            reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) |
                dot + n.embedding[i] * $q[i]
            ) /
            (
                sqrt(reduce(norm_n = 0.0, i IN range(0, size(n.embedding)-1) |
                norm_n + n.embedding[i] * n.embedding[i]
                )) *
                sqrt(reduce(norm_q = 0.0, i IN range(0, size($q)-1) |
                norm_q + $q[i] * $q[i]
                ))
            ) AS score
            WHERE score > 0.5  // Zwiększony próg minimalny
            RETURN elementId(n) AS node_id, n, score
            ORDER BY score DESC
            LIMIT $k
            '''

        with self.driver.session() as session:
            base = session.run(vec_query, q=query_embedding, k=top_k).data()
        
        results = []
        for record in base:
            node = record['n']
            score = record['score']
            node_id = record['node_id']
            results.append({
                'id': node_id,
                'data': node,
                'score': score
            })
            
        # 2. Graph expansion - find related images through SIMILAR_TO relationships
        graph_query = '''
        MATCH (n)-[r:SIMILAR_TO]->(nbr:Chunk)
        WHERE elementId(n) = $node_id AND nbr.type = 'image'
        RETURN elementId(nbr) AS nbr_id, nbr, r.weight AS score
        ORDER BY score DESC
        LIMIT $m
        '''
        
        # Track visited nodes to avoid re-exploring
        visited = set(item['id'] for item in results)
        
        for item in list(results):
            nbrs = self.driver.session().run(
                graph_query,
                node_id=item['id'],
                m=expand_k
            ).data()
            
            for rec in nbrs:
                nbr = rec['nbr']
                nbr_id = rec['nbr_id']
                weight = rec['score']
                
                if nbr_id not in visited:
                    visited.add(nbr_id)
                    results.append({
                        'id': nbr_id,
                        'data': nbr,
                        'score': weight * 0.9  # Slightly discount graph-expanded results
                    })

        # 3. Rerank and dedupe
        merged = {}
        for r in results:
            nid = r['id']
            if nid in merged:
                merged[nid]['score'] = (merged[nid]['score'] + r['score']) / 2
            else:
                merged[nid] = r
                
        # Sort by score
        final = sorted(merged.values(), key=lambda x: x['score'], reverse=True)
        return final[:top_k]

    def search_by_text(self, text_query, text_embedder, top_k=5):
        """
        Search for images using a text query.
        
        Args:
            text_query: Text description to search for
            text_embedder: TextEmbedder instance to generate text embeddings
            top_k: Number of results to return
            
        Returns:
            List of image results
        """
        # Get embedding for the text query
        query_embedding = text_embedder.get_text_embedding(text_query).tolist()
        
        # Use the embedding to search for similar images
        return self.retrieve(query_embedding, top_k=top_k)
        
    def get_image_info(self, image_path):
        """
        Retrieve metadata about an image from the database.
        
        Args:
            image_path: Path of the image to retrieve info about
            
        Returns:
            Dictionary with image metadata or None if not found
        """
        query = '''
        MATCH (n:Chunk {path: $path, type: 'image'})
        RETURN n
        LIMIT 1
        '''
        
        with self.driver.session() as session:
            result = session.run(query, path=image_path).single()
            
        if result:
            return result['n']
        return None
class GraphCleaner:
    """
    Utility class for cleaning relationships in the Neo4j database.
    Provides methods to delete all relationships or specific relationship types.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    def delete_all_relationships(self):
        """
        Delete ALL relationships in the database.
        Use with caution as this operation cannot be undone.
        """
        query = """
        MATCH ()-[r]->()
        DELETE r
        """
        with self.driver.session() as session:
            result = session.run(query)
            return result.consume().counters.relationships_deleted

    def delete_specific_relationships(self, relationship_types=None):
        """
        Delete relationships of specific types.
        
        Args:
            relationship_types: List of relationship type names to delete.
                            If None, defaults to known semantic relationship types.
        
        Returns:
            Number of relationships deleted
        """
        if relationship_types is None:
            # Default to the semantic relationship types used in the application
            relationship_types = [
                "SIMILAR_TO", 
                "IMAGE_SIMILAR", 
                "IMAGE_ILLUSTRATES", 
                "TEXT_ILLUSTRATED_BY"
            ]
        
        # Poprawiony sposób budowania wzorca relacji - dwukropek tylko na początku
        rel_pattern = ":" + "|".join(relationship_types)
        
        query = f"""
        MATCH ()-[r {rel_pattern}]->()
        DELETE r
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            return result.consume().counters.relationships_deleted

    def get_relationship_counts(self):
        """
        Get counts of all relationship types in the database.
        
        Returns:
            Dictionary mapping relationship types to their counts
        """
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        WITH relationshipType
        MATCH ()-[r]->() 
        WHERE type(r) = relationshipType
        RETURN relationshipType, count(r) as count
        ORDER BY count DESC
        """
        
        with self.driver.session() as session:
            results = session.run(query).data()
            return {rec["relationshipType"]: rec["count"] for rec in results}
            
    def get_node_counts(self):
        """
        Get counts of nodes by label in the database.
        
        Returns:
            Dictionary mapping node labels to their counts
        """
        query = """
        CALL db.labels() YIELD label
        WITH label
        MATCH (n) 
        WHERE label in labels(n)
        RETURN label, count(n) as count
        ORDER BY count DESC
        """
        
        with self.driver.session() as session:
            results = session.run(query).data()
            return {rec["label"]: rec["count"] for rec in results}

    def reset_database(self, keep_nodes=True):
        """
        Reset the database by deleting all relationships and optionally all nodes.
        
        Args:
            keep_nodes: If True, keeps all nodes but deletes relationships.
                       If False, deletes both nodes and relationships.
        
        Returns:
            Dictionary with counts of deleted objects
        """
        deleted_rels = self.delete_all_relationships()
        deleted_nodes = 0
        
        if not keep_nodes:
            query = "MATCH (n) DELETE n"
            with self.driver.session() as session:
                result = session.run(query)
                deleted_nodes = result.consume().counters.nodes_deleted
        
        return {
            "relationships_deleted": deleted_rels,
            "nodes_deleted": deleted_nodes
        }