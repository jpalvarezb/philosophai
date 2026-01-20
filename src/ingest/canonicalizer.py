"""Entity canonicalization via lemmatization."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage import DuckDBStorage


class EntityCanonicalizer:
    """Canonicalize entities via lemmatization to merge plurals/variants."""

    def __init__(self, storage: "DuckDBStorage"):
        self.storage = storage
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            import spacy
            print("🔧 Loading spaCy model...")
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return self._nlp

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
