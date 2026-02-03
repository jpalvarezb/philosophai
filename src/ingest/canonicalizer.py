"""Entity and predicate canonicalization via lemmatization + embeddings."""
from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from ..storage import DuckDBStorage


# Predicate synonym groups for high-confidence merges
PREDICATE_SYNONYMS = {
    "is": ["are", "be", "was", "were", "being", "am"],
    "has": ["have", "possesses", "possess", "owns", "own", "holds", "hold"],
    "contains": ["includes", "include", "comprises", "comprise", "consists of",
                 "incorporates", "incorporate", "encompasses", "encompass"],
    "causes": ["cause", "produces", "produce", "creates", "create", "generates",
               "results in", "leads to", "brings about", "gives rise to"],
    "requires": ["require", "needs", "need", "depends on", "necessitates", "demands"],
    "belongs to": ["belong to", "is part of", "is member of", "falls under"],
    "derived from": ["derives from", "comes from", "originates from", "stems from", "arises from"],
    "called": ["named", "known as", "referred to as", "termed", "designated", "means"],
    "similar to": ["like", "resembles", "is like", "comparable to", "akin to"],
    "gives": ["give", "provides", "provide", "offers", "supplies", "grants", "bestows"],
    "makes": ["make", "constructs", "builds", "forms", "fashions", "fabricates"],
    "lacks": ["lack", "is without", "has no", "missing", "does not have"],
    "believes": ["believe", "thinks", "think", "considers", "regards", "holds that"],
    "knows": ["know", "understands", "understand", "recognizes", "is aware of"],
    "located in": ["located at", "found in", "situated in", "exists in", "resides in"],
    "associated with": ["related to", "connected to", "linked to", "tied to"],
    "used for": ["used as", "employed for", "utilized for", "serves as"],
    "becomes": ["become", "turns into", "changes into", "transforms into", "evolves into"],
}

# Build reverse lookup
PREDICATE_TO_CANON = {}
for canon, variants in PREDICATE_SYNONYMS.items():
    PREDICATE_TO_CANON[canon] = canon
    for v in variants:
        PREDICATE_TO_CANON[v] = canon


class EntityCanonicalizer:
    """Canonicalize entities and predicates via lemmatization + semantic embeddings."""

    def __init__(
        self,
        storage: "DuckDBStorage",
        llm_client: "OpenAI | None" = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.storage = storage
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            import spacy
            print("🔧 Loading spaCy model...")
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return self._nlp

    def get_embeddings_batch(self, texts: list[str], batch_size: int = 500) -> np.ndarray:
        """Get embeddings for a list of texts."""
        if not self.llm_client:
            raise ValueError("LLM client required for embeddings")
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t else " " for t in texts[i:i + batch_size]]
            response = self.llm_client.embeddings.create(
                input=batch,
                model=self.embedding_model,
            )
            all_embeddings.extend([e.embedding for e in response.data])
        return np.array(all_embeddings)

    def cluster_by_similarity(
        self,
        terms: list[str],
        embeddings: np.ndarray,
        threshold: float = 0.92,
    ) -> dict[str, str]:
        """Cluster terms by embedding similarity, return term -> canonical mapping."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        n = len(terms)
        if n == 0:
            return {}
        
        sim_matrix = cosine_similarity(embeddings)
        
        # Greedy clustering
        visited = set()
        clusters = []
        
        for i in range(n):
            if i in visited:
                continue
            cluster = {i}
            queue = [i]
            while queue:
                curr = queue.pop(0)
                for j in range(n):
                    if j not in visited and j not in cluster and sim_matrix[curr, j] >= threshold:
                        cluster.add(j)
                        queue.append(j)
            visited.update(cluster)
            clusters.append(cluster)
        
        # Canonical = shortest term (most general)
        term_to_canon = {}
        for cluster in clusters:
            cluster_terms = [terms[i] for i in cluster]
            canonical = min(cluster_terms, key=len)
            for t in cluster_terms:
                term_to_canon[t] = canonical
        
        return term_to_canon

    def canonicalize(
        self,
        source_table: str = "normalized_triples_clean",
        target_table: str = "normalized_triples_clean_canon",
        batch_size: int = 1000,
    ) -> dict:
        """
        Canonicalize entities in triples via lemmatization.
        
        Creates:
        - entity_canon_map: original -> canonical mapping
        - target_table: triples with canonicalized entities
        
        Args:
            source_table: Table with cleaned triples
            target_table: Table to create with canonicalized triples
            batch_size: Batch size for spaCy processing
        
        Returns:
            Dict with statistics
        """
        con = self.storage.con
        
        # 1. Get all unique entities
        print("📊 Extracting unique entities...")
        entities_df = con.execute(f"""
            SELECT DISTINCT entity FROM (
                SELECT subject_norm as entity FROM {source_table}
                UNION
                SELECT object_norm as entity FROM {source_table} 
                WHERE object_norm IS NOT NULL AND object_norm != ''
            )
        """).fetchdf()
        
        entities = entities_df['entity'].tolist()
        print(f"   Found {len(entities):,} unique entities")
        
        # 2. Lemmatize all entities
        print("🔄 Lemmatizing entities...")
        entity_to_canon = {}
        
        from tqdm import tqdm
        for i in tqdm(range(0, len(entities), batch_size), desc="Lemmatizing"):
            batch = entities[i:i+batch_size]
            docs = list(self.nlp.pipe([e.lower() if e else "" for e in batch]))
            
            for entity, doc in zip(batch, docs):
                if entity:
                    lemmas = " ".join([token.lemma_ for token in doc])
                    entity_to_canon[entity] = lemmas
        
        # 3. Check merge statistics
        unique_canons = set(entity_to_canon.values())
        merged_count = len(entities) - len(unique_canons)
        print(f"   Original entities: {len(entities):,}")
        print(f"   Canonical forms:   {len(unique_canons):,}")
        print(f"   Merged:            {merged_count:,} ({100*merged_count/len(entities):.1f}%)")
        
        # 4. Create mapping table
        print("📝 Creating entity_canon_map table...")
        import pandas as pd
        map_df = pd.DataFrame([
            {"entity_orig": k, "entity_canon": v} 
            for k, v in entity_to_canon.items()
        ])
        
        con.execute("DROP TABLE IF EXISTS entity_canon_map")
        con.execute("CREATE TABLE entity_canon_map AS SELECT * FROM map_df")
        
        # 5. Create canonicalized triples table
        print(f"🔧 Creating {target_table}...")
        con.execute(f"DROP TABLE IF EXISTS {target_table}")
        con.execute(f"""
            CREATE TABLE {target_table} AS
            SELECT 
                t.triple_id,
                t.text_id,
                t.chunk_id,
                t.subject_raw,
                t.predicate_raw,
                t.object_raw,
                COALESCE(sm.entity_canon, t.subject_norm) as subject_norm,
                t.predicate_norm,
                COALESCE(om.entity_canon, t.object_norm) as object_norm,
                t.flags_json,
                t.model_name,
                t.run_id,
                t.created_at,
                COALESCE(sm.entity_canon, t.subject_norm) as subject_canon_id,
                t.predicate_norm as predicate_canon_id,
                COALESCE(om.entity_canon, t.object_norm) as object_canon_id
            FROM {source_table} t
            LEFT JOIN entity_canon_map sm ON t.subject_norm = sm.entity_orig
            LEFT JOIN entity_canon_map om ON t.object_norm = om.entity_orig
        """)
        
        # 6. Verify
        count = con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
        print(f"✅ Created {target_table} with {count:,} triples")
        
        # Show sample merges
        print("\n📋 Sample merges:")
        samples = con.execute("""
            SELECT entity_orig, entity_canon
            FROM entity_canon_map
            WHERE entity_orig != entity_canon
            LIMIT 15
        """).fetchdf()
        for _, row in samples.iterrows():
            print(f'   "{row["entity_orig"]}" → "{row["entity_canon"]}"')
        
        return {
            "original_entities": len(entities),
            "canonical_entities": len(unique_canons),
            "merged": merged_count,
            "triples": count,
        }

    def get_canonical_form(self, entity: str) -> str:
        """Get the canonical form of a single entity."""
        doc = self.nlp(entity.lower())
        return " ".join([token.lemma_ for token in doc])

    def lookup_canonical(self, entity: str) -> str | None:
        """Look up canonical form from the mapping table."""
        result = self.storage.con.execute(
            "SELECT entity_canon FROM entity_canon_map WHERE entity_orig = ?",
            [entity],
        ).fetchone()
        return result[0] if result else None

    def canonicalize_predicates(
        self,
        source_table: str = "normalized_triples_clean_canon",
        similarity_threshold: float = 0.88,
        min_frequency: int = 2,
    ) -> dict:
        """
        Canonicalize predicates using synonym list + embedding similarity.
        
        Args:
            source_table: Table with predicates to canonicalize
            similarity_threshold: Min cosine similarity to merge
            min_frequency: Only embed predicates with >= this frequency
        """
        con = self.storage.con
        
        print("\n" + "=" * 50)
        print("PREDICATE CANONICALIZATION")
        print("=" * 50)
        
        # 1. Get all unique predicates with frequency
        print("📊 Extracting predicates...")
        pred_df = con.execute(f"""
            SELECT predicate_norm as predicate, COUNT(*) as cnt
            FROM {source_table}
            WHERE predicate_norm IS NOT NULL AND predicate_norm != ''
            GROUP BY predicate_norm
            ORDER BY cnt DESC
        """).fetchdf()
        
        all_predicates = pred_df['predicate'].tolist()
        print(f"   Found {len(all_predicates):,} unique predicates")
        
        # 2. First pass: apply synonym list + lemmatization
        print("🔄 Applying synonym list + lemmatization...")
        pred_to_canon = {}
        from tqdm import tqdm
        
        for pred in tqdm(all_predicates, desc="Processing"):
            pred_lower = pred.lower().strip()
            
            # Try synonym list first
            if pred_lower in PREDICATE_TO_CANON:
                pred_to_canon[pred] = PREDICATE_TO_CANON[pred_lower]
                continue
            
            # Lemmatize
            doc = self.nlp(pred_lower)
            lemma = " ".join([t.lemma_ for t in doc])
            
            # Check if lemma is in synonym list
            if lemma in PREDICATE_TO_CANON:
                pred_to_canon[pred] = PREDICATE_TO_CANON[lemma]
            else:
                pred_to_canon[pred] = lemma
        
        # Stats after first pass
        unique_after_lemma = len(set(pred_to_canon.values()))
        print(f"   After lemmatization: {len(all_predicates):,} → {unique_after_lemma:,}")
        
        # 3. Second pass: embedding clustering for frequent predicates
        if self.llm_client:
            high_freq_preds = pred_df[pred_df['cnt'] >= min_frequency]['predicate'].tolist()
            
            if len(high_freq_preds) > 1:
                print(f"🧠 Computing embeddings for {len(high_freq_preds):,} frequent predicates...")
                
                # Get canonical forms for high-freq predicates
                canon_forms = list(set(pred_to_canon[p] for p in high_freq_preds))
                
                if len(canon_forms) > 1:
                    embeddings = self.get_embeddings_batch(canon_forms)
                    
                    print(f"🔗 Clustering (threshold={similarity_threshold})...")
                    canon_to_merged = self.cluster_by_similarity(
                        canon_forms, embeddings, similarity_threshold
                    )
                    
                    # Apply to all predicates
                    for pred, canon in pred_to_canon.items():
                        if canon in canon_to_merged:
                            pred_to_canon[pred] = canon_to_merged[canon]
        
        # Final stats
        unique_canons = set(pred_to_canon.values())
        merged = len(all_predicates) - len(unique_canons)
        print(f"   Final: {len(all_predicates):,} → {len(unique_canons):,}")
        print(f"   Merged: {merged:,} predicates ({100*merged/len(all_predicates):.1f}%)")
        
        # 4. Save mapping and update table
        print("💾 Saving predicate_canon_map...")
        import pandas as pd
        map_df = pd.DataFrame([
            {"predicate_orig": k, "predicate_canon": v}
            for k, v in pred_to_canon.items()
        ])
        con.execute("DROP TABLE IF EXISTS predicate_canon_map")
        con.execute("CREATE TABLE predicate_canon_map AS SELECT * FROM map_df")
        
        print(f"🔧 Updating {source_table}...")
        con.execute(f"""
            UPDATE {source_table} t
            SET predicate_canon_id = (
                SELECT predicate_canon FROM predicate_canon_map m
                WHERE m.predicate_orig = t.predicate_norm
            )
        """)
        
        # Show top merges
        print("\n📋 Top predicate clusters:")
        clusters = defaultdict(list)
        for orig, canon in pred_to_canon.items():
            if orig.lower() != canon:
                clusters[canon].append(orig)
        
        for canon, variants in sorted(clusters.items(), key=lambda x: -len(x[1]))[:8]:
            print(f'   "{canon}" ← {variants[:5]}')
        
        return {"original": len(all_predicates), "canonical": len(unique_canons), "merged": merged}

    def canonicalize_entities_semantic(
        self,
        source_table: str = "normalized_triples_clean_canon",
        similarity_threshold: float = 0.92,
        min_frequency: int = 2,
        max_entities: int = 10000,
    ) -> dict:
        """
        Further canonicalize entities using embedding similarity.
        Run AFTER basic lemmatization canonicalize().
        
        Args:
            source_table: Table with entities
            similarity_threshold: Min cosine similarity to merge
            min_frequency: Only embed entities with >= this frequency
            max_entities: Cap on entities to embed (cost control)
        """
        if not self.llm_client:
            print("⚠️ LLM client required for semantic entity canonicalization")
            return {"merged": 0}
        
        con = self.storage.con
        
        print("\n" + "=" * 50)
        print("SEMANTIC ENTITY CANONICALIZATION")
        print("=" * 50)
        
        # Get frequent entities
        print("📊 Extracting frequent entities...")
        entity_df = con.execute(f"""
            SELECT entity, SUM(cnt) as total FROM (
                SELECT subject_canon_id as entity, COUNT(*) as cnt FROM {source_table} GROUP BY 1
                UNION ALL
                SELECT object_canon_id as entity, COUNT(*) as cnt FROM {source_table}
                WHERE object_canon_id IS NOT NULL GROUP BY 1
            )
            GROUP BY entity
            HAVING SUM(cnt) >= {min_frequency}
            ORDER BY total DESC
            LIMIT {max_entities}
        """).fetchdf()
        
        entities = entity_df['entity'].tolist()
        print(f"   Found {len(entities):,} entities (freq >= {min_frequency})")
        
        if len(entities) < 2:
            return {"merged": 0}
        
        # Get embeddings
        print("🧠 Computing entity embeddings...")
        embeddings = self.get_embeddings_batch(entities)
        
        # Cluster
        print(f"🔗 Clustering (threshold={similarity_threshold})...")
        entity_to_merged = self.cluster_by_similarity(
            entities, embeddings, similarity_threshold
        )
        
        # Stats
        unique_canons = set(entity_to_merged.values())
        merged = len(entities) - len(unique_canons)
        print(f"   Clustered: {len(entities):,} → {len(unique_canons):,}")
        print(f"   Merged: {merged:,} entities")
        
        if merged == 0:
            return {"merged": 0}
        
        # Update entity_canon_map
        print("💾 Updating entity_canon_map...")
        existing = con.execute("SELECT entity_orig, entity_canon FROM entity_canon_map").fetchdf()
        existing_dict = dict(zip(existing['entity_orig'], existing['entity_canon']))
        
        # Remap: if old canonical → new canonical, update all that pointed to old
        for old_canon, new_canon in entity_to_merged.items():
            if old_canon != new_canon:
                for orig, canon in list(existing_dict.items()):
                    if canon == old_canon:
                        existing_dict[orig] = new_canon
        
        import pandas as pd
        map_df = pd.DataFrame([
            {"entity_orig": k, "entity_canon": v}
            for k, v in existing_dict.items()
        ])
        con.execute("DROP TABLE IF EXISTS entity_canon_map")
        con.execute("CREATE TABLE entity_canon_map AS SELECT * FROM map_df")
        
        # Update source table
        print(f"🔧 Updating {source_table}...")
        con.execute(f"""
            UPDATE {source_table} t
            SET 
                subject_canon_id = COALESCE(
                    (SELECT entity_canon FROM entity_canon_map m WHERE m.entity_orig = t.subject_norm),
                    t.subject_canon_id
                ),
                object_canon_id = COALESCE(
                    (SELECT entity_canon FROM entity_canon_map m WHERE m.entity_orig = t.object_norm),
                    t.object_canon_id
                )
        """)
        
        # Show sample merges
        print("\n📋 Sample entity clusters:")
        clusters = defaultdict(list)
        for orig, canon in entity_to_merged.items():
            if orig != canon:
                clusters[canon].append(orig)
        
        for canon, variants in sorted(clusters.items(), key=lambda x: -len(x[1]))[:8]:
            print(f'   "{canon}" ← {variants[:5]}')
        
        return {"original": len(entities), "canonical": len(unique_canons), "merged": merged}
    def dedupe_subject_object_pairs(
        self,
        table: str = "normalized_triples_clean_canon",
    ) -> dict:
        """
        Keep only the highest-support predicate per (subject, object) pair.

        Support = row count for that subject/object/predicate. Ties break
        lexicographically on predicate_canon_id for determinism.
        """
        con = self.storage.con
        print("\n" + "=" * 60)
        print("DEDUP SUBJECT–OBJECT PAIRS")
        print("=" * 60)

        try:
            before_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            before_pairs = con.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT subject_canon_id, object_canon_id
                    FROM {table}
                    WHERE object_canon_id IS NOT NULL AND object_canon_id != ''
                    GROUP BY 1, 2
                )
                """
            ).fetchone()[0]
        except Exception as e:
            print(f"❌ Could not read from {table}: {e}")
            return {"deduped": False, "error": str(e)}

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS
            WITH pred_support AS (
                SELECT
                    subject_canon_id,
                    object_canon_id,
                    predicate_canon_id,
                    COUNT(*) AS support
                FROM {table}
                WHERE object_canon_id IS NOT NULL AND object_canon_id != ''
                GROUP BY 1, 2, 3
            ),
            ranked AS (
                SELECT
                    t.*,
                    ps.support,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.subject_canon_id, t.object_canon_id
                        ORDER BY ps.support DESC NULLS LAST, t.predicate_canon_id
                    ) AS pred_rank
                FROM {table} t
                LEFT JOIN pred_support ps
                  ON t.subject_canon_id = ps.subject_canon_id
                 AND t.object_canon_id = ps.object_canon_id
                 AND t.predicate_canon_id = ps.predicate_canon_id
            )
            SELECT * EXCLUDE(pred_rank, support)
            FROM ranked
            WHERE (object_canon_id IS NULL OR object_canon_id = '' OR pred_rank = 1)
            """
        )

        after_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        after_pairs = con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT subject_canon_id, object_canon_id
                FROM {table}
                WHERE object_canon_id IS NOT NULL AND object_canon_id != ''
                GROUP BY 1, 2
            )
            """
        ).fetchone()[0]

        removed_rows = before_rows - after_rows
        multi_pairs_removed = con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT subject_canon_id, object_canon_id
                FROM {table}
                WHERE object_canon_id IS NOT NULL AND object_canon_id != ''
                GROUP BY 1, 2
                HAVING COUNT(DISTINCT predicate_canon_id) > 1
            )
            """
        ).fetchone()[0]

        print(f"Rows: {before_rows:,} → {after_rows:,} (removed {removed_rows:,})")
        print(f"Subject–object pairs: {before_pairs:,} → {after_pairs:,}")
        print(f"Pairs with >1 predicate remaining: {multi_pairs_removed:,}")

        return {
            "deduped": True,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "removed_rows": removed_rows,
            "pairs": after_pairs,
            "multi_pred_pairs_remaining": multi_pairs_removed,
        }

    def canonicalize_all(
        self,
        source_table: str = "normalized_triples_clean",
        target_table: str = "normalized_triples_clean_canon",
        predicate_threshold: float = 0.88,
        entity_threshold: float = 0.92,
    ) -> dict:
        """
        Full canonicalization pipeline:
        1. Lemmatize entities → create target_table
        2. Canonicalize predicates (synonym list + embeddings)
        3. Canonicalize entities further (embeddings)
        4. Show final edge weight distribution
        """
        print("\n" + "=" * 60)
        print("FULL CANONICALIZATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Basic entity lemmatization
        print("\n--- Step 1: Entity Lemmatization ---")
        entity_stats = self.canonicalize(source_table, target_table)
        
        # Step 2: Predicate canonicalization
        print("\n--- Step 2: Predicate Canonicalization ---")
        pred_stats = self.canonicalize_predicates(
            target_table, similarity_threshold=predicate_threshold
        )
        
        # Step 3: Semantic entity merging
        print("\n--- Step 3: Semantic Entity Merging ---")
        sem_stats = self.canonicalize_entities_semantic(
            target_table, similarity_threshold=entity_threshold
        )

        # Step 4: Deduplicate subject-object pairs to a single predicate
        print("\n--- Step 4: Deduplicate Subject/Object Pairs ---")
        dedup_stats = self.dedupe_subject_object_pairs(target_table)
        
        # Final edge weight distribution
        con = self.storage.con
        print("\n" + "=" * 60)
        print("FINAL EDGE WEIGHT DISTRIBUTION")
        print("=" * 60)
        
        edge_stats = con.execute(f"""
            SELECT weight, COUNT(*) as cnt FROM (
                SELECT subject_canon_id, predicate_canon_id, object_canon_id,
                       COUNT(DISTINCT chunk_id) as weight
                FROM {target_table}
                GROUP BY 1, 2, 3
            )
            GROUP BY weight ORDER BY weight
        """).fetchdf()
        
        for _, row in edge_stats.head(10).iterrows():
            print(f"   weight={int(row['weight'])}: {int(row['cnt']):,} edges")
        
        total = edge_stats['cnt'].sum()
        gt1 = edge_stats[edge_stats['weight'] > 1]['cnt'].sum()
        print(f"\n   Total edges: {total:,}")
        print(f"   Edges with weight > 1: {gt1:,} ({100*gt1/total:.1f}%)")
        
        return {
            "entity_lemmatization": entity_stats,
            "predicate_canonicalization": pred_stats,
            "semantic_entity_merging": sem_stats,
            "dedup": dedup_stats,
        }
