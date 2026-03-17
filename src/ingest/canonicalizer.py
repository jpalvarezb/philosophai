"""Entity and predicate canonicalization via lemmatization + embeddings."""
from __future__ import annotations

import json
from collections import defaultdict, deque
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from .config import IngestConfig

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

# Prepositions that carry distinct semantic roles (agent, locative, topic, etc.)
# Different (verb_stem, prep) pairs must not merge unless whitelisted below.
PREPOSITION_ROLE_SET = frozenset({
    "by", "in", "on", "at", "to", "for", "from", "with", "about", "as",
    "into", "onto", "upon", "within", "without", "during", "after", "before",
})

# Pairs of prepositions treated as equivalent for predicate merging (symmetric).
# E.g. "written about" and "wrote on" both indicate topic.
PREPOSITION_SYNONYM_PAIRS = frozenset({
    ("about", "on"),   # topic: "written about" ≈ "wrote on"
    ("on", "about"),
})


class EntityCanonicalizer:
    """Canonicalize entities and predicates via lemmatization + semantic embeddings."""

    def __init__(
        self,
        storage: "DuckDBStorage",
        llm_client: "OpenAI | None" = None,
        embedding_model: str = "text-embedding-3-small",
        judge_model: str = "gpt-5-mini",
        config: IngestConfig | None = None,
    ):
        self.storage = storage
        self.llm_client = llm_client
        self.config = config or IngestConfig()
        self.embedding_model = embedding_model or self.config.embedding_model
        self.judge_model = judge_model or self.config.judge_model
        self._nlp = None
        self._ner_nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            import spacy
            print("🔧 Loading spaCy model...")
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return self._nlp

    @property
    def ner_nlp(self):
        """Lazy load spaCy pipeline with NER enabled."""
        if self._ner_nlp is None:
            import spacy

            print("🔧 Loading spaCy NER model...")
            self._ner_nlp = spacy.load("en_core_web_sm", disable=["parser"])
        return self._ner_nlp

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

    @staticmethod
    def _pair_key(left: str, right: str) -> tuple[str, str]:
        """Create an order-independent key for a pair of terms."""
        return tuple(sorted((left, right)))

    @staticmethod
    def _normalize_pair_set(
        pairs: set[tuple[str, str]] | None,
    ) -> set[tuple[str, str]]:
        """Normalize pair collections to sorted tuple keys."""
        if not pairs:
            return set()
        return {tuple(sorted(pair)) for pair in pairs}

    @staticmethod
    def _compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix for embeddings."""
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(embeddings)

    def _should_link_pair(
        self,
        left_idx: int,
        right_idx: int,
        terms: list[str],
        sim_matrix: np.ndarray,
        threshold: float,
        blocked_pairs: set[tuple[str, str]],
        forced_pairs: set[tuple[str, str]],
    ) -> bool:
        """Return whether two terms should be linked in the merge graph."""
        pair_key = self._pair_key(terms[left_idx], terms[right_idx])
        if pair_key in forced_pairs:
            return True
        if pair_key in blocked_pairs:
            return False
        return bool(sim_matrix[left_idx, right_idx] >= threshold)

    def _connected_components(
        self,
        indices: list[int],
        terms: list[str],
        sim_matrix: np.ndarray,
        threshold: float,
        blocked_pairs: set[tuple[str, str]],
        forced_pairs: set[tuple[str, str]],
    ) -> list[set[int]]:
        """Build connected components for a thresholded similarity graph."""
        if not indices:
            return []

        pending = list(indices)
        visited: set[int] = set()
        clusters: list[set[int]] = []

        for idx in pending:
            if idx in visited:
                continue

            cluster = {idx}
            queue = deque([idx])
            visited.add(idx)

            while queue:
                current = queue.popleft()
                for candidate in pending:
                    if candidate in visited:
                        continue
                    if not self._should_link_pair(
                        current,
                        candidate,
                        terms,
                        sim_matrix,
                        threshold,
                        blocked_pairs,
                        forced_pairs,
                    ):
                        continue
                    visited.add(candidate)
                    cluster.add(candidate)
                    queue.append(candidate)

            clusters.append(cluster)

        return clusters

    def _cluster_min_pair_similarity(
        self,
        cluster: set[int],
        terms: list[str],
        sim_matrix: np.ndarray,
        forced_pairs: set[tuple[str, str]],
    ) -> float:
        """Return the minimum pairwise similarity inside a cluster."""
        if len(cluster) < 2:
            return 1.0

        min_similarity = 1.0
        saw_unforced_pair = False
        for left_idx, right_idx in combinations(sorted(cluster), 2):
            if self._pair_key(terms[left_idx], terms[right_idx]) in forced_pairs:
                continue
            saw_unforced_pair = True
            min_similarity = min(min_similarity, float(sim_matrix[left_idx, right_idx]))

        return min_similarity if saw_unforced_pair else 1.0

    def _refine_cluster(
        self,
        cluster: set[int],
        terms: list[str],
        sim_matrix: np.ndarray,
        threshold: float,
        blocked_pairs: set[tuple[str, str]],
        forced_pairs: set[tuple[str, str]],
        validation_margin: float,
        refinement_step: float,
        max_refinement_threshold: float,
    ) -> list[set[int]]:
        """Split weakly connected clusters by progressively tightening the threshold."""
        if len(cluster) < 3:
            return [cluster]

        cohesion_floor = max(-1.0, threshold - validation_margin)
        min_similarity = self._cluster_min_pair_similarity(
            cluster, terms, sim_matrix, forced_pairs
        )
        if min_similarity >= cohesion_floor:
            return [cluster]

        stricter_threshold = min(max_refinement_threshold, threshold + refinement_step)
        if stricter_threshold <= threshold:
            return [cluster]

        subclusters = self._connected_components(
            indices=sorted(cluster),
            terms=terms,
            sim_matrix=sim_matrix,
            threshold=stricter_threshold,
            blocked_pairs=blocked_pairs,
            forced_pairs=forced_pairs,
        )
        if len(subclusters) == 1:
            if stricter_threshold >= max_refinement_threshold:
                return [cluster]
            return self._refine_cluster(
                cluster=cluster,
                terms=terms,
                sim_matrix=sim_matrix,
                threshold=stricter_threshold,
                blocked_pairs=blocked_pairs,
                forced_pairs=forced_pairs,
                validation_margin=validation_margin,
                refinement_step=refinement_step,
                max_refinement_threshold=max_refinement_threshold,
            )

        refined_clusters: list[set[int]] = []
        for subcluster in subclusters:
            refined_clusters.extend(
                self._refine_cluster(
                    cluster=subcluster,
                    terms=terms,
                    sim_matrix=sim_matrix,
                    threshold=stricter_threshold,
                    blocked_pairs=blocked_pairs,
                    forced_pairs=forced_pairs,
                    validation_margin=validation_margin,
                    refinement_step=refinement_step,
                    max_refinement_threshold=max_refinement_threshold,
                )
            )
        return refined_clusters

    def _select_cluster_canonical(
        self,
        cluster: set[int],
        terms: list[str],
        embeddings: np.ndarray,
        frequencies: dict[str, int] | None = None,
    ) -> str:
        """Pick the centroid-nearest term, with deterministic tie-breakers."""
        cluster_indices = sorted(cluster)
        if len(cluster_indices) == 1:
            return terms[cluster_indices[0]]

        cluster_embeddings = embeddings[cluster_indices]
        centroid = cluster_embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            centroid_scores = np.zeros(len(cluster_indices))
        else:
            cluster_norms = np.linalg.norm(cluster_embeddings, axis=1)
            safe_norms = np.where(cluster_norms == 0, 1.0, cluster_norms)
            centroid_scores = (cluster_embeddings @ centroid) / (safe_norms * centroid_norm)

        best_score = float(np.max(centroid_scores))
        candidate_indices = [
            cluster_indices[pos]
            for pos, score in enumerate(centroid_scores)
            if np.isclose(score, best_score)
        ]
        if frequencies:
            max_frequency = max(frequencies.get(terms[idx], 0) for idx in candidate_indices)
            candidate_indices = [
                idx
                for idx in candidate_indices
                if frequencies.get(terms[idx], 0) == max_frequency
            ]

        return max(
            (terms[idx] for idx in candidate_indices),
            key=lambda term: (len(term), term.lower()),
        )

    def _predicate_core_tokens(self, predicate: str) -> set[str]:
        """Extract lemma tokens for predicate direction checks."""
        skip_tokens = {"is", "are", "was", "were", "be", "been", "being", "by", "from", "of"}
        return {
            token.lemma_.lower()
            for token in self.nlp(predicate.lower())
            if token.is_alpha and token.lemma_.lower() not in skip_tokens
        }

    def is_inverse_predicate_pair(self, left: str, right: str) -> bool:
        """Detect likely inverse predicate phrasing such as active vs passive voice."""
        passive_starts = ("is ", "are ", "was ", "were ", "be ", "been ", "being ")

        left_norm = left.lower().strip()
        right_norm = right.lower().strip()
        left_passive = left_norm.startswith(passive_starts) and " by" in left_norm
        right_passive = right_norm.startswith(passive_starts) and " by" in right_norm

        if left_passive == right_passive:
            return False

        left_core = self._predicate_core_tokens(left_norm)
        right_core = self._predicate_core_tokens(right_norm)
        return bool(left_core and right_core and left_core == right_core)

    def _get_predicate_verb_preposition(self, predicate: str) -> tuple[str, str] | None:
        """
        Extract (verb_stem, trailing_preposition) when the predicate has a clear
        [verb][prep] pattern. E.g. "written by" -> ("write", "by"), "wrote on" -> ("write", "on").
        Returns None if there is no trailing preposition or no clear verb before it.
        """
        doc = self.nlp(predicate.lower().strip())
        if len(doc) < 2:
            return None
        tokens = list(doc)
        # Find last token that is a preposition in our role set
        prep_idx = None
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i].lemma_.lower() in PREPOSITION_ROLE_SET:
                prep_idx = i
                break
        if prep_idx is None or prep_idx == 0:
            return None
        verb_lemma = tokens[prep_idx - 1].lemma_.lower()
        prep_lemma = tokens[prep_idx].lemma_.lower()
        # Don't treat preposition as verb (e.g. "in in" edge case)
        if verb_lemma in PREPOSITION_ROLE_SET:
            return None
        return (verb_lemma, prep_lemma)

    def has_conflicting_preposition_roles(self, left: str, right: str) -> bool:
        """
        True when both predicates have the same verb stem but different trailing
        prepositions that are not whitelisted as synonyms. Blocks e.g. "written by"
        from merging with "written in" or "wrote on".
        """
        left_vp = self._get_predicate_verb_preposition(left)
        right_vp = self._get_predicate_verb_preposition(right)
        if left_vp is None or right_vp is None:
            return False
        (v1, p1), (v2, p2) = left_vp, right_vp
        if v1 != v2:
            return False
        if p1 == p2:
            return False
        pair = (p1, p2) if p1 < p2 else (p2, p1)
        if pair in PREPOSITION_SYNONYM_PAIRS:
            return False
        return True

    @staticmethod
    def _extract_full_span_entity_label(doc) -> str | None:
        """Return the NER label only when the entire phrase is a named entity."""
        named_entity_labels = {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "NORP",
            "EVENT",
            "FAC",
            "LAW",
            "WORK_OF_ART",
        }
        for ent in doc.ents:
            if ent.start == 0 and ent.end == len(doc) and ent.label_ in named_entity_labels:
                return ent.label_
        return None

    def _load_entity_named_flags(self) -> dict[str, bool]:
        """Load named-entity flags aggregated to current canonical entities."""
        con = self.storage.con
        try:
            rows = con.execute(
                """
                SELECT
                    m.entity_canon,
                    MAX(CASE WHEN n.is_named_entity THEN 1 ELSE 0 END) AS is_named_entity
                FROM entity_canon_map m
                LEFT JOIN entity_ner_map n
                  ON m.entity_orig = n.entity_orig
                GROUP BY 1
                """
            ).fetchall()
        except Exception:
            return {}

        return {row[0]: bool(row[1]) for row in rows}

    def _parse_json_content(self, content: str) -> dict:
        """Parse a JSON object from model output."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        return json.loads(text)

    def judge_same_entity_pairs(
        self,
        candidate_pairs: list[dict[str, str | float]],
        batch_size: int = 25,
    ) -> set[tuple[str, str]]:
        """Use an LLM to decide which ambiguous entity pairs should merge."""
        if not self.llm_client or not candidate_pairs:
            return set()

        approved_pairs: set[tuple[str, str]] = set()
        system_prompt = (
            "You are deciding whether two entity labels from a philosophical knowledge graph "
            "refer to the same underlying entity. Only answer true when they are clear aliases, "
            "variant spellings, abbreviations, or equivalent titles. Do not merge related-but-distinct "
            "concepts, people in the same category, or broader/narrower terms. Return strict JSON only."
        )

        for start in range(0, len(candidate_pairs), batch_size):
            batch = candidate_pairs[start : start + batch_size]
            payload = {"pairs": batch}
            response = self.llm_client.chat.completions.create(
                model=self.judge_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "For each pair, decide if the two labels are the same entity.\n"
                            "Return JSON with this schema: "
                            '{"pairs":[{"left":"...", "right":"...", "same_entity": true, "reason": "..."}]}\n\n'
                            f"{json.dumps(payload, ensure_ascii=True)}"
                        ),
                    },
                ],
            )
            content = response.choices[0].message.content or "{}"
            parsed = self._parse_json_content(content)
            for item in parsed.get("pairs", []):
                if not item.get("same_entity"):
                    continue
                left = str(item.get("left", "")).strip()
                right = str(item.get("right", "")).strip()
                if not left or not right:
                    continue
                approved_pairs.add(self._pair_key(left, right))

        return approved_pairs

    def cluster_by_similarity(
        self,
        terms: list[str],
        embeddings: np.ndarray,
        threshold: float = 0.92,
        *,
        blocked_pairs: set[tuple[str, str]] | None = None,
        forced_pairs: set[tuple[str, str]] | None = None,
        frequencies: dict[str, int] | None = None,
        sim_matrix: np.ndarray | None = None,
        validation_margin: float = 0.08,
        refinement_step: float = 0.03,
        max_refinement_threshold: float = 0.98,
    ) -> dict[str, str]:
        """Cluster terms by embedding similarity, return term -> canonical mapping."""
        n = len(terms)
        if n == 0:
            return {}

        blocked_pairs = self._normalize_pair_set(blocked_pairs)
        forced_pairs = self._normalize_pair_set(forced_pairs)
        if sim_matrix is None:
            sim_matrix = self._compute_similarity_matrix(embeddings)

        initial_clusters = self._connected_components(
            indices=list(range(n)),
            terms=terms,
            sim_matrix=sim_matrix,
            threshold=threshold,
            blocked_pairs=blocked_pairs,
            forced_pairs=forced_pairs,
        )

        clusters: list[set[int]] = []
        for cluster in initial_clusters:
            clusters.extend(
                self._refine_cluster(
                    cluster=cluster,
                    terms=terms,
                    sim_matrix=sim_matrix,
                    threshold=threshold,
                    blocked_pairs=blocked_pairs,
                    forced_pairs=forced_pairs,
                    validation_margin=validation_margin,
                    refinement_step=refinement_step,
                    max_refinement_threshold=max_refinement_threshold,
                )
            )

        term_to_canon = {}
        for cluster in clusters:
            canonical = self._select_cluster_canonical(
                cluster=cluster,
                terms=terms,
                embeddings=embeddings,
                frequencies=frequencies,
            )
            for idx in cluster:
                term_to_canon[terms[idx]] = canonical

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
        entity_ner_rows: list[dict[str, object]] = []
        
        from tqdm import tqdm
        for i in tqdm(range(0, len(entities), batch_size), desc="Lemmatizing"):
            batch = entities[i:i+batch_size]
            docs = list(self.nlp.pipe([e.lower() if e else "" for e in batch]))
            ner_docs = list(self.ner_nlp.pipe([e if e else "" for e in batch]))
            
            for entity, doc, ner_doc in zip(batch, docs, ner_docs):
                if entity:
                    lemmas = " ".join([token.lemma_ for token in doc])
                    entity_to_canon[entity] = lemmas
                    ner_label = self._extract_full_span_entity_label(ner_doc)
                    entity_ner_rows.append(
                        {
                            "entity_orig": entity,
                            "ner_label": ner_label,
                            "is_named_entity": bool(ner_label),
                        }
                    )
        
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

        ner_df = pd.DataFrame(entity_ner_rows)
        if ner_df.empty:
            ner_df = pd.DataFrame(
                columns=["entity_orig", "ner_label", "is_named_entity"]
            )
        con.execute("DROP TABLE IF EXISTS entity_ner_map")
        con.execute("CREATE TABLE entity_ner_map AS SELECT * FROM ner_df")
        
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
                pred_frequencies = dict(zip(pred_df["predicate"], pred_df["cnt"]))
                canon_frequencies: defaultdict[str, int] = defaultdict(int)
                for pred, canon in pred_to_canon.items():
                    canon_frequencies[canon] += int(pred_frequencies.get(pred, 0))
                
                if len(canon_forms) > 1:
                    embeddings = self.get_embeddings_batch(canon_forms)
                    blocked_pairs = {
                        self._pair_key(left, right)
                        for left, right in combinations(canon_forms, 2)
                        if self.is_inverse_predicate_pair(left, right)
                        or self.has_conflicting_preposition_roles(left, right)
                    }
                    
                    print(f"🔗 Clustering (threshold={similarity_threshold})...")
                    canon_to_merged = self.cluster_by_similarity(
                        canon_forms,
                        embeddings,
                        similarity_threshold,
                        blocked_pairs=blocked_pairs,
                        frequencies=dict(canon_frequencies),
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
        ambiguous_margin: float = 0.07,
        samejudge_batch_size: int = 25,
        max_samejudge_pairs: int = 500,
        named_entity_threshold_boost: float = 0.05,
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
        entity_frequencies = {
            row["entity"]: int(row["total"])
            for _, row in entity_df.iterrows()
        }
        print(f"   Found {len(entities):,} entities (freq >= {min_frequency})")
        
        if len(entities) < 2:
            return {"merged": 0}
        
        # Get embeddings
        print("🧠 Computing entity embeddings...")
        embeddings = self.get_embeddings_batch(entities)
        sim_matrix = self._compute_similarity_matrix(embeddings)

        named_entity_flags = self._load_entity_named_flags()
        blocked_pairs: set[tuple[str, str]] = set()
        ambiguous_pairs: list[dict[str, str | float | int | bool]] = []
        for left_idx in range(len(entities)):
            left = entities[left_idx]
            left_named = named_entity_flags.get(left, False)
            for right_idx in range(left_idx + 1, len(entities)):
                right = entities[right_idx]
                similarity = float(sim_matrix[left_idx, right_idx])
                both_named = left_named and named_entity_flags.get(right, False)
                effective_threshold = similarity_threshold + (
                    named_entity_threshold_boost if both_named else 0.0
                )
                soft_lower = max(0.0, effective_threshold - ambiguous_margin)
                pair_key = self._pair_key(left, right)

                if both_named and similarity < effective_threshold:
                    blocked_pairs.add(pair_key)

                if soft_lower <= similarity < effective_threshold:
                    ambiguous_pairs.append(
                        {
                            "left": left,
                            "right": right,
                            "similarity": round(similarity, 4),
                            "left_frequency": entity_frequencies.get(left, 0),
                            "right_frequency": entity_frequencies.get(right, 0),
                            "both_named_entities": both_named,
                        }
                    )

        if len(ambiguous_pairs) > max_samejudge_pairs:
            ambiguous_pairs.sort(
                key=lambda item: (
                    -float(item["similarity"]),
                    -(int(item["left_frequency"]) + int(item["right_frequency"])),
                    str(item["left"]),
                    str(item["right"]),
                )
            )
            print(
                f"⚠️ Truncating SameJudge candidates from {len(ambiguous_pairs):,} "
                f"to {max_samejudge_pairs:,} most promising pairs"
            )
            ambiguous_pairs = ambiguous_pairs[:max_samejudge_pairs]

        forced_pairs = self.judge_same_entity_pairs(
            ambiguous_pairs,
            batch_size=samejudge_batch_size,
        )
        if forced_pairs:
            print(f"🤝 SameJudge approved {len(forced_pairs):,} ambiguous merges")
        
        # Cluster
        print(f"🔗 Clustering (threshold={similarity_threshold})...")
        entity_to_merged = self.cluster_by_similarity(
            entities,
            embeddings,
            similarity_threshold,
            blocked_pairs=blocked_pairs,
            forced_pairs=forced_pairs,
            frequencies=entity_frequencies,
            sim_matrix=sim_matrix,
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
        predicate_threshold: float | None = None,
        entity_threshold: float | None = None,
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

        predicate_threshold = (
            predicate_threshold
            if predicate_threshold is not None
            else self.config.predicate_threshold
        )
        entity_threshold = (
            entity_threshold if entity_threshold is not None else self.config.entity_threshold
        )
        
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
