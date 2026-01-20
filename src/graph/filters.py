"""Graph filtering utilities for stop-entities, hubs, and generic predicates."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


# Generic concepts that shouldn't anchor traversal
STOP_ENTITY_LABELS = frozenset({
    # Pronouns / determiners
    "it", "they", "we", "he", "she", "this", "that", "these", "those",
    "what", "which", "who", "whom", "one", "other", "another", "some",
    # Generic nouns
    "thing", "something", "nothing", "everything", "anything",
    "being", "entity", "object", "item", "element", "unit",
    "part", "whole", "portion", "aspect", "feature",
    "kind", "sort", "type", "class", "category", "species", "genus",
    "way", "manner", "mode", "method", "means",
    "case", "instance", "example", "situation", "circumstance",
    "fact", "point", "matter", "issue", "question", "problem",
    "time", "place", "moment", "period", "state", "condition",
    # Quantifiers
    "all", "many", "few", "several", "most", "none", "each", "every",
    "same", "different", "similar", "various", "certain",
    # Adjectives used nominally
    "true", "false", "good", "bad", "great", "small", "large",
    "first", "second", "third", "last", "former", "latter",
    "new", "old", "young", "like",
    # Extraction artifacts
    "unmixe", "etc", "ie", "eg", "cf", "viz",
})

# Generic predicates that add little semantic value (weight = 0.2)
GENERIC_PREDICATES = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "does", "do", "did",
    "relates_to", "related_to", "relates", "related",
    "associated_with", "associated", "linked_to", "linked",
    "connected_to", "connected", "involves", "involved",
    "concerns", "pertains_to", "pertains",
    "is_a", "is_an", "is_type_of", "type_of",
    "has_property", "has_attribute", "has_quality",
    "includes", "contains", "comprises",
    "said", "says", "called", "named", "known_as",
    "like", "such as", "for example",
})

# High-value predicates for philosophical/ontological reasoning (weight = 1.5)
VALUABLE_PREDICATES = frozenset({
    # Ontology / metaphysics
    "is actuality of", "is essence of", "is form of", "is matter of",
    "is substance of", "is attribute of", "is mode of",
    "is cause of", "causes", "is caused by", "produces", "is produced by",
    "depends on", "is dependent on", "presupposes", "entails",
    "is necessary for", "is sufficient for", "is condition of",
    # Psychology / soul
    "animates", "is animated by", "moves", "is moved by",
    "perceives", "is perceived by", "thinks", "wills", "desires",
    "is faculty of", "has faculty", "is power of", "has power",
    "is part of soul", "is function of",
    # Relations
    "is contrary to", "is opposite of", "contradicts",
    "is prior to", "is posterior to", "is simultaneous with",
    "is greater than", "is less than", "is equal to",
    "is identical to", "differs from", "is distinct from",
    # Definitions
    "is defined as", "means", "signifies", "denotes",
})

# Blocked predicates - too generic to be useful (weight = 0)
BLOCKED_PREDICATES = frozenset({
    "and", "or", "but", "with", "without", "in", "on", "at", "to", "from",
    "see", "cf", "viz", "etc", "ie", "eg",
})

# Common function words for label quality check
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "also", "now", "even", "still", "already", "always",
    "and", "but", "or", "if", "when", "where", "while", "because",
    "that", "which", "who", "what", "this", "these", "those",
    "it", "its", "they", "their", "them", "we", "our", "us",
    "he", "she", "his", "her", "him",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
    "despite", "although", "however", "therefore", "thus",
})


class GraphFilters:
    """
    Configurable filters for graph traversal and seeding.
    
    Usage:
        filters = GraphFilters(graph, hub_threshold_pct=0.01)
        if filters.is_valid_seed(entity_id):
            seeds.append(entity_id)
    """
    
    def __init__(
        self,
        graph: "nx.MultiDiGraph",
        hub_threshold_pct: float = 0.01,
        min_degree: int = 1,
        stop_labels: frozenset[str] | None = None,
        generic_predicates: frozenset[str] | None = None,
    ):
        """
        Args:
            graph: The knowledge graph
            hub_threshold_pct: Nodes with degree > this % of total are hubs
            min_degree: Minimum degree for valid seeds
            stop_labels: Custom stop entity labels (defaults to STOP_ENTITY_LABELS)
            generic_predicates: Custom generic predicates (defaults to GENERIC_PREDICATES)
        """
        self.graph = graph
        self.min_degree = min_degree
        self.stop_labels = stop_labels or STOP_ENTITY_LABELS
        self.generic_predicates = generic_predicates or GENERIC_PREDICATES
        
        # Compute hub threshold
        n_nodes = graph.number_of_nodes()
        self.hub_threshold = max(10, int(n_nodes * hub_threshold_pct))
    
    def is_stop_entity(self, entity_id: str) -> bool:
        """Check if entity is a generic stop concept."""
        label = self.graph.nodes.get(entity_id, {}).get("label", entity_id)
        return self._label_is_stop(label)
    
    def _label_is_stop(self, label: str) -> bool:
        """Check if label matches stop patterns."""
        label_lower = label.lower().strip()
        # Exact match
        if label_lower in self.stop_labels:
            return True
        # Single-word check for very short labels
        if len(label_lower) <= 3 and label_lower not in {"god", "man", "one", "two"}:
            return True
        return False
    
    def is_hub(self, entity_id: str) -> bool:
        """Check if entity is a high-degree hub."""
        if entity_id not in self.graph:
            return False
        degree = self.graph.degree(entity_id)
        return degree > self.hub_threshold
    
    def has_min_degree(self, entity_id: str) -> bool:
        """Check if entity meets minimum degree requirement."""
        if entity_id not in self.graph:
            return False
        return self.graph.degree(entity_id) >= self.min_degree
    
    def is_generic_predicate(self, predicate: str) -> bool:
        """Check if predicate is too generic to be useful."""
        pred_lower = predicate.lower().strip()
        return pred_lower in self.generic_predicates
    
    def is_valid_seed(self, entity_id: str) -> bool:
        """Check if entity is valid for seeding traversal."""
        if entity_id not in self.graph:
            return False
        if self.is_stop_entity(entity_id):
            return False
        if self.is_hub(entity_id):
            return False
        if not self.has_min_degree(entity_id):
            return False
        # Check label quality
        label = self.graph.nodes.get(entity_id, {}).get("label", entity_id)
        if self.is_low_quality_label(label):
            return False
        return True
    
    def is_valid_expansion(self, entity_id: str) -> bool:
        """Check if entity is valid for traversal expansion (less strict than seed)."""
        if entity_id not in self.graph:
            return False
        if self.is_stop_entity(entity_id):
            return False
        # Check label quality (filter extraction artifacts)
        label = self.graph.nodes.get(entity_id, {}).get("label", entity_id)
        if self.is_low_quality_label(label):
            return False
        # Allow hubs during expansion but deprioritize them
        return True
    
    def predicate_weight(self, predicate: str) -> float:
        """
        Return weight for predicate based on semantic value.
        
        Returns:
            0.0 for blocked predicates
            0.2 for generic predicates
            1.0 for normal predicates
            1.5 for high-value philosophical predicates
        """
        pred_lower = predicate.lower().strip()
        
        # Check blocked first
        if pred_lower in BLOCKED_PREDICATES:
            return 0.0
        
        # Check generic BEFORE valuable ("is" should be generic, not match "is actuality of")
        if pred_lower in self.generic_predicates:
            return 0.2
        
        # Check valuable predicates - exact match only, or pred contains valuable
        if pred_lower in VALUABLE_PREDICATES:
            return 1.5
        for vp in VALUABLE_PREDICATES:
            # Only match if valuable predicate is contained in pred (not vice versa)
            # This prevents "is" from matching "is actuality of"
            if vp in pred_lower:
                return 1.5
        
        return 1.0
    
    def edge_weight_modifier(self, predicate: str) -> float:
        """Alias for predicate_weight for backward compatibility."""
        return self.predicate_weight(predicate)
    
    def is_low_quality_label(self, label: str) -> bool:
        """
        Check if entity label is likely an extraction artifact.
        
        Filters:
        - Contains unusual punctuation (-, :, etc.)
        - Mostly function words
        - Very long multi-word phrases
        - "X and Y" compound phrases
        - Starts with/contains prepositions indicating clause fragments
        - Starts with determiners (another, some, other)
        """
        import re
        
        label_lower = label.lower().strip()
        
        # Contains unusual punctuation patterns
        if re.search(r'[-:;"\(\)\[\]]', label_lower):
            return True
        
        # Split into words
        words = re.findall(r'[a-z]+', label_lower)
        if not words:
            return True
        
        # Too many words (likely a clause)
        if len(words) > 4:
            return True
        
        # "X and Y" compound patterns (often extraction artifacts)
        if len(words) >= 3 and 'and' in words:
            return True
        
        # Prepositions/conjunctions that indicate clause fragments
        clause_markers = {'without', 'upon', 'despite', 'before', 'after',
                          'during', 'through', 'under', 'over', 'within',
                          'between', 'among', 'towards', 'against',
                          'from', 'into', 'onto', 'or', 'nor'}
        
        # Starts with preposition
        if words[0] in clause_markers:
            return True
        
        # Contains preposition in middle (e.g., "body without impulsion", "reason within soul")
        if len(words) >= 3:
            for w in words[1:-1]:  # Check middle words
                if w in clause_markers:
                    return True
        
        # Starts with determiner/quantifier (another, some, other, etc.)
        determiner_starts = {'another', 'some', 'other', 'any', 'each', 'every',
                             'certain', 'such', 'many', 'few', 'several'}
        if words[0] in determiner_starts:
            return True
        
        # Mostly function words
        function_count = sum(1 for w in words if w in FUNCTION_WORDS)
        if len(words) > 1 and function_count / len(words) > 0.5:
            return True
        
        # Starts with function word and has multiple words
        if len(words) > 1 and words[0] in FUNCTION_WORDS:
            return True
        
        return False
    
    def filter_seeds(self, entity_ids: list[str]) -> list[str]:
        """Filter a list of entity IDs to valid seeds."""
        return [eid for eid in entity_ids if self.is_valid_seed(eid)]
