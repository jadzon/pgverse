from neo4j import GraphDatabase

from neo4j import GraphDatabase

class GraphBuilder:
    """
    Builds a semantic knowledge graph in Neo4j by creating SIMILAR_TO relationships
    between TextChunk and ImageNode nodes based on cosine similarity.
    """
    def __init__(self, uri, user, password, similarity_threshold=0.7):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.threshold = similarity_threshold

    def close(self):
        self.driver.close()

    def create_relationships(self):
        """
        Create SIMILAR_TO relationships with weight and last_used timestamp
        between all pairs of TextChunk and ImageNode.
        """
        # Using pure Cypher implementation of cosine similarity instead of GDS
        query = '''
        MATCH (a),(b)
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
        ON MATCH SET  r.weight = sim
        '''
        with self.driver.session() as session:
            session.run(query, threshold=self.threshold)

class HybridRetriever:
    """
    Performs hybrid retrieval combining vector search and graph expansion.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def retrieve(self, query_embedding, top_k=5, expand_k=3):
        """
        Returns a list of node dicts (id, text/url, score)
        """
        # 1. vector-based retrieval
        vec_query = '''
        MATCH (n)
        WHERE exists(n.embedding)
        WITH n, gds.alpha.similarity.cosine(n.embedding, $q) AS score
        WHERE score > 0
        RETURN n, score
        ORDER BY score DESC
        LIMIT $k
        '''
        with self.driver.session() as session:
            base = session.run(vec_query, q=query_embedding, k=top_k).data()
        results = []
        for record in base:
            node = record['n']
            score = record['score']
            results.append({
                'id': node.id,
                'data': dict(node),
                'score': score
            })
        # 2. graph expansion
        graph_query = '''
        MATCH (n)-[r:SIMILAR_TO]->(nbr)
        WHERE id(n) = $node_id
        RETURN nbr, r.weight AS score
        ORDER BY score DESC
        LIMIT $m
        '''
        for item in list(results):
            nbrs = self.driver.session().run(
                graph_query,
                node_id=item['id'],
                m=expand_k
            ).data()
            for rec in nbrs:
                nbr = rec['nbr']
                weight = rec['score']
                results.append({
                    'id': nbr.id,
                    'data': dict(nbr),
                    'score': weight
                })
        # 3. rerank and dedupe
        # average scores if duplicates
        merged = {}
        for r in results:
            nid = r['id']
            if nid in merged:
                merged[nid]['score'] = (merged[nid]['score'] + r['score']) / 2
            else:
                merged[nid] = r
        # sort
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
        Apply exponential decay to relationship weights.
        """
        query = '''
        MATCH ()-[r:SIMILAR_TO]->()
        SET r.weight = r.weight * exp(-$decay * (timestamp() - r.last_used))
        '''
        with self.driver.session() as session:
            session.run(query, decay=self.decay)

    def prune_old(self):
        """
        Remove relationships not used within max_age.
        """
        query = '''
        MATCH ()-[r:SIMILAR_TO]->()
        WHERE r.last_used < timestamp() - $max_age
        DELETE r
        '''
        with self.driver.session() as session:
            session.run(query, max_age=self.max_age)

    def run_maintenance(self):
        self.decay_weights()
        self.prune_old()
