"""Embedding-based conceptness scoring for entity quality filtering.

Computes conceptness(e) = cohesion(e) * support(e) where:
- cohesion(e) = average cosine similarity between emb(e) and emb(neighbor) for top-N neighbors
- support(e) = log1p(mention_count) + log1p(weighted_degree)

Entities with high conceptness are stable philosophical concepts.
Entities with low conceptness are likely extraction artifacts (e.g., "soul cottage").
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from ..storage import DuckDBStorage


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class ConceptnessScorer:
    """
    Compute and store conceptness scores for all entities.
    
    Usage:
        scorer = ConceptnessScorer(storage, graph)
        scorer.compute_all(top_n_neighbors=10)
        # Now entity_conceptness table is populated
    """
    
    def __init__(
        self,
        storage: "DuckDBStorage",
        graph: "nx.MultiDiGraph",
    ):
        self.storage = storage
        self.graph = graph
        self._embeddings: dict[str, list[float]] = {}
    
    def _ensure_table(self):
        """Create entity_conceptness table if not exists."""
        self.storage.con.execute("""
            CREATE TABLE IF NOT EXISTS entity_conceptness (
                entity_id VARCHAR PRIMARY KEY,
                label VARCHAR,
                cohesion DOUBLE,
                support DOUBLE,
                conceptness DOUBLE,
                neighbor_count INTEGER,
                weighted_degree INTEGER,
                mention_count INTEGER
            )
        """)
    
    def _load_embeddings(self) -> dict[str, list[float]]:
        """Load entity embeddings from DB."""
        if self._embeddings:
            return self._embeddings
        
        try:
            rows = self.storage.con.execute("""
                SELECT entity_canon, embedding 
                FROM entity_embeddings
                WHERE embedding IS NOT NULL
            """).fetchall()
            self._embeddings = {row[0]: row[1] for row in rows}
            print(f"✅ Loaded {len(self._embeddings):,} entity embeddings")
        except Exception as e:
            print(f"⚠️ Could not load entity embeddings: {e}")
            self._embeddings = {}
        
        return self._embeddings
    
    def _get_mention_counts(self) -> dict[str, int]:
        """Get mention count per entity from triples."""
        rows = self.storage.con.execute("""
            SELECT entity_id, SUM(cnt) as total
            FROM (
                SELECT subject_canon_id as entity_id, COUNT(*) as cnt
                FROM normalized_triples_clean_canon
                GROUP BY 1
                UNION ALL
                SELECT object_canon_id as entity_id, COUNT(*) as cnt
                FROM normalized_triples_clean_canon
                GROUP BY 1
            )
            GROUP BY entity_id
        """).fetchall()
        return {row[0]: row[1] for row in rows}
    
    def _get_top_neighbors(self, entity_id: str, top_n: int = 10) -> list[tuple[str, int]]:
        """Get top-N neighbors by edge weight."""
        if entity_id not in self.graph:
            return []
        
        neighbors: dict[str, int] = {}
        
        # Outgoing edges
        for neighbor in self.graph.successors(entity_id):
            for _, edge_data in self.graph[entity_id][neighbor].items():
                weight = edge_data.get("weight", 1)
                neighbors[neighbor] = neighbors.get(neighbor, 0) + weight
        
        # Incoming edges
        for neighbor in self.graph.predecessors(entity_id):
            for _, edge_data in self.graph[neighbor][entity_id].items():
                weight = edge_data.get("weight", 1)
                neighbors[neighbor] = neighbors.get(neighbor, 0) + weight
        
        # Sort by weight desc
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: -x[1])
        return sorted_neighbors[:top_n]
    
    def compute_cohesion(self, entity_id: str, top_n: int = 10) -> tuple[float, int]:
        """
        Compute embedding cohesion for an entity.
        
        Returns:
            (cohesion_score, neighbor_count)
        """
        embeddings = self._load_embeddings()
        
        if entity_id not in embeddings:
            return 0.0, 0
        
        entity_emb = embeddings[entity_id]
        neighbors = self._get_top_neighbors(entity_id, top_n)
        
        if not neighbors:
            return 0.0, 0
        
        # Compute average cosine similarity to neighbors
        similarities = []
        for neighbor_id, _ in neighbors:
            if neighbor_id in embeddings:
                sim = cosine_similarity(entity_emb, embeddings[neighbor_id])
                similarities.append(sim)
        
        if not similarities:
            return 0.0, 0
        
        cohesion = sum(similarities) / len(similarities)
        return cohesion, len(similarities)
    
    def compute_support(self, entity_id: str, mention_counts: dict[str, int]) -> tuple[float, int, int]:
        """
        Compute support score for an entity.
        
        Returns:
            (support_score, weighted_degree, mention_count)
        """
        # Weighted degree from graph
        weighted_degree = 0
        if entity_id in self.graph:
            for neighbor in self.graph.successors(entity_id):
                for _, edge_data in self.graph[entity_id][neighbor].items():
                    weighted_degree += edge_data.get("weight", 1)
            for neighbor in self.graph.predecessors(entity_id):
                for _, edge_data in self.graph[neighbor][entity_id].items():
                    weighted_degree += edge_data.get("weight", 1)
        
        mention_count = mention_counts.get(entity_id, 0)
        
        support = math.log1p(mention_count) + math.log1p(weighted_degree)
        return support, weighted_degree, mention_count
    
    def compute_all(
        self,
        top_n_neighbors: int = 10,
        batch_size: int = 1000,
    ) -> dict:
        """
        Compute conceptness scores for all entities and store in DB.
        
        Args:
            top_n_neighbors: Number of neighbors for cohesion calculation
            batch_size: Insert batch size
        
        Returns:
            Statistics dict
        """
        from tqdm import tqdm
        
        self._ensure_table()
        self._load_embeddings()
        mention_counts = self._get_mention_counts()
        
        # Get all entities from graph
        entities = list(self.graph.nodes())
        print(f"📊 Computing conceptness for {len(entities):,} entities...")
        
        # Clear existing
        self.storage.con.execute("DELETE FROM entity_conceptness")
        
        results = []
        for entity_id in tqdm(entities, desc="Computing conceptness"):
            label = self.graph.nodes[entity_id].get("label", entity_id)
            
            cohesion, neighbor_count = self.compute_cohesion(entity_id, top_n_neighbors)
            support, weighted_degree, mention_count = self.compute_support(entity_id, mention_counts)
            
            # Conceptness = cohesion * support
            # Normalize cohesion to 0-1 range (cosine sim can be negative)
            normalized_cohesion = max(0, cohesion)
            conceptness = normalized_cohesion * support
            
            results.append((
                entity_id, label, cohesion, support, conceptness,
                neighbor_count, weighted_degree, mention_count
            ))
            
            # Batch insert
            if len(results) >= batch_size:
                self._insert_batch(results)
                results = []
        
        # Final batch
        if results:
            self._insert_batch(results)
        
        # Compute statistics
        stats = self.storage.con.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(conceptness) as avg_conceptness,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY conceptness) as p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY conceptness) as p50,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY conceptness) as p75,
                MIN(conceptness) as min_conceptness,
                MAX(conceptness) as max_conceptness
            FROM entity_conceptness
        """).fetchone()
        
        print(f"✅ Computed conceptness for {stats[0]:,} entities")
        print(f"   Mean: {stats[1]:.3f}, Median: {stats[3]:.3f}")
        print(f"   P25: {stats[2]:.3f}, P75: {stats[4]:.3f}")
        print(f"   Range: [{stats[5]:.3f}, {stats[6]:.3f}]")
        
        return {
            "total": stats[0],
            "avg": stats[1],
            "p25": stats[2],
            "median": stats[3],
            "p75": stats[4],
            "min": stats[5],
            "max": stats[6],
        }
    
    def _insert_batch(self, results: list[tuple]):
        """Insert batch of results."""
        self.storage.con.executemany("""
            INSERT INTO entity_conceptness 
            (entity_id, label, cohesion, support, conceptness, 
             neighbor_count, weighted_degree, mention_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, results)
    
    def get_threshold(self, percentile: float = 0.30) -> float:
        """
        Get conceptness threshold at given percentile.
        
        Args:
            percentile: Fraction (0-1) for threshold
        
        Returns:
            Conceptness value at that percentile
        """
        result = self.storage.con.execute(f"""
            SELECT PERCENTILE_CONT({percentile}) WITHIN GROUP (ORDER BY conceptness)
            FROM entity_conceptness
        """).fetchone()
        return result[0] if result else 0.0
    
    def sample_by_conceptness(self, n: int = 20, low: bool = True) -> list[tuple[str, float]]:
        """
        Sample entities by conceptness for inspection.
        
        Args:
            n: Number to sample
            low: If True, sample lowest; if False, sample highest
        
        Returns:
            List of (label, conceptness) tuples
        """
        order = "ASC" if low else "DESC"
        rows = self.storage.con.execute(f"""
            SELECT label, conceptness
            FROM entity_conceptness
            ORDER BY conceptness {order}
            LIMIT ?
        """, [n]).fetchall()
        return rows


def compute_conceptness_scores(
    storage: "DuckDBStorage",
    graph: "nx.MultiDiGraph",
    top_n_neighbors: int = 10,
) -> dict:
    """
    Convenience function to compute all conceptness scores.
    
    Returns statistics dict.
    """
    scorer = ConceptnessScorer(storage, graph)
    return scorer.compute_all(top_n_neighbors=top_n_neighbors)
