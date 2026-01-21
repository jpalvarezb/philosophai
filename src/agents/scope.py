"""Scope extraction and filtering for author/work/tradition/domain constraints.

Enables queries like "What does Aristotle say about X?" to restrict retrieval
to only Aristotle's works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage import DuckDBStorage


class ScopeViolationError(Exception):
    """Raised when strict scope constraints are violated.
    
    This is a real exception, not an assertion, so it cannot be
    stripped by Python's -O optimization flag.
    """
    
    def __init__(self, stage: str, message: str, examples: list | None = None):
        self.stage = stage
        self.message = message
        self.examples = examples or []
        super().__init__(f"Strict scope violation at {stage}: {message}")


# Valid schema values (must match files table exactly)
VALID_AUTHORS = frozenset({
    "Al-Ghazali", "Anonymous (Maya)", "Aristotle", "Arthur Schopenhauer",
    "Attributed to Buddha", "Attributed to Manu", "Attributed to Muhammad",
    "Augustine", "Averroes", "Benedict of Nursia", "Bronislaw Malinowski",
    "Confucius", "David Hume", "Djéli Mamoudou Kouyaté (griot tradition)",
    "Dogen", "Epictetus", "Epicurus", "Friedrich Nietzsche", "Gautama",
    "Herodotus", "Ibn Khaldun", "Immanuel Kant", "Ishvara Krishna",
    "John Calvin", "John Locke", "John Stuart Mill", "Kautilya", "Laozi",
    "Lucretius", "Maimonides", "Marcus Aurelius", "Margaret Mead",
    "Meiji Government", "Mencius", "Mozi", "Mumon Ekai", "Murasaki Shikibu",
    "Nagarjuna", "Nitobe Inazo", "Pali Canon", "Patanjali", "Plato",
    "Plotinus", "Plutarch", "Polybius", "Ramanuja", "Sei Shonagon",
    "Seneca", "Shankara", "Shantideva", "Svatmarama", "Theodor Herzl",
    "Thomas à Kempis", "Thucydides", "Tokugawa Shogunate", "Unknown",
    "Valmiki", "Various", "Vyasa", "Xunzi", "Yamamoto Tsunetomo",
    "Yuanwu / Engo", "Zhuangzi", "Émile Durkheim",
})

VALID_TRADITIONS = frozenset({
    "Greek–Hellenistic", "Indian", "Japanese", "Modern European",
    "Chinese", "Christian", "Islamic", "Jewish",
    "Anthropology (empirical)", "African Indigenous", "Americas Indigenous",
})

VALID_DOMAINS = frozenset({
    "Ethics", "Metaphysics", "History", "Theology", "Anthropology", "Epistemology",
})

# Author name variations for fuzzy matching
AUTHOR_ALIASES = {
    # Greek philosophers
    "aristotle": "Aristotle",
    "plato": "Plato",
    "socrates": "Plato",  # Socrates' views through Plato's dialogues
    "epictetus": "Epictetus",
    "epicurus": "Epicurus",
    "marcus aurelius": "Marcus Aurelius",
    "aurelius": "Marcus Aurelius",
    "seneca": "Seneca",
    "plotinus": "Plotinus",
    "lucretius": "Lucretius",
    "herodotus": "Herodotus",
    "thucydides": "Thucydides",
    "polybius": "Polybius",
    "plutarch": "Plutarch",
    # Modern European
    "kant": "Immanuel Kant",
    "immanuel kant": "Immanuel Kant",
    "hume": "David Hume",
    "david hume": "David Hume",
    "nietzsche": "Friedrich Nietzsche",
    "friedrich nietzsche": "Friedrich Nietzsche",
    "schopenhauer": "Arthur Schopenhauer",
    "arthur schopenhauer": "Arthur Schopenhauer",
    "locke": "John Locke",
    "john locke": "John Locke",
    "mill": "John Stuart Mill",
    "john stuart mill": "John Stuart Mill",
    "durkheim": "Émile Durkheim",
    # Christian
    "augustine": "Augustine",
    "st augustine": "Augustine",
    "saint augustine": "Augustine",
    "calvin": "John Calvin",
    "john calvin": "John Calvin",
    "thomas à kempis": "Thomas à Kempis",
    "thomas a kempis": "Thomas à Kempis",
    "benedict": "Benedict of Nursia",
    # Islamic
    "averroes": "Averroes",
    "ibn rushd": "Averroes",
    "al-ghazali": "Al-Ghazali",
    "ghazali": "Al-Ghazali",
    "ibn khaldun": "Ibn Khaldun",
    # Jewish
    "maimonides": "Maimonides",
    "rambam": "Maimonides",
    "herzl": "Theodor Herzl",
    # Indian
    "vyasa": "Vyasa",
    "shankara": "Shankara",
    "adi shankara": "Shankara",
    "shankaracharya": "Shankara",
    "ramanuja": "Ramanuja",
    "patanjali": "Patanjali",
    "nagarjuna": "Nagarjuna",
    "gautama": "Gautama",
    "kautilya": "Kautilya",
    "chanakya": "Kautilya",
    "shantideva": "Shantideva",
    "valmiki": "Valmiki",
    # Chinese
    "confucius": "Confucius",
    "kong qiu": "Confucius",
    "mencius": "Mencius",
    "mengzi": "Mencius",
    "laozi": "Laozi",
    "lao tzu": "Laozi",
    "zhuangzi": "Zhuangzi",
    "chuang tzu": "Zhuangzi",
    "mozi": "Mozi",
    "mo tzu": "Mozi",
    "xunzi": "Xunzi",
    "hsun tzu": "Xunzi",
    # Japanese
    "dogen": "Dogen",
    "mumon": "Mumon Ekai",
    "nitobe": "Nitobe Inazo",
    # Buddha
    "buddha": "Attributed to Buddha",
    "gautama buddha": "Attributed to Buddha",
    "siddhartha": "Attributed to Buddha",
}

# Work title aliases
WORK_ALIASES = {
    # Aristotle
    "de anima": "On the Soul (De Anima)",
    "on the soul": "On the Soul (De Anima)",
    "nicomachean ethics": "Nicomachean Ethics",
    "ethics": "Nicomachean Ethics",  # Ambiguous but common
    "physics": "Physics",
    "metaphysics": "Metaphysics",
    "politics": "Politics",
    "categories": "Categories",
    "posterior analytics": "Posterior Analytics",
    # Plato
    "republic": "Republic",
    "the republic": "Republic",
    "timaeus": "Timaeus",
    "laws": "Laws",
    "phaedo": "Apology, Crito, and Phaedo",
    "apology": "Apology, Crito, and Phaedo",
    "theaetetus": "Theaetetus",
    "sophist": "Sophist",
    "parmenides": "Parmenides",
    # Kant
    "critique of pure reason": "Critique of Pure Reason",
    "first critique": "Critique of Pure Reason",
    "critique of practical reason": "Critique of Practical Reason",
    "second critique": "Critique of Practical Reason",
    # Hume
    "treatise": "A Treatise of Human Nature",
    "treatise of human nature": "A Treatise of Human Nature",
    "enquiry concerning human understanding": "Enquiry Concerning Human Understanding",
    # Augustine
    "confessions": "Confessions",
    "city of god": "City of God",
    # Indian
    "bhagavad gita": "Bhagavad Gita",
    "gita": "Bhagavad Gita",
    "upanishads": "Upanishads (principal)",
    "yoga sutras": "Yoga Sutras",
    "brahma sutras": "Brahma Sutras",
    # Chinese
    "analects": "Analects",
    "dao de jing": "Dao De Jing",
    "tao te ching": "Dao De Jing",
    # Buddhist
    "dhammapada": "Dhammapada",
}

# Tradition aliases
TRADITION_ALIASES = {
    "greek": "Greek–Hellenistic",
    "hellenistic": "Greek–Hellenistic",
    "ancient greek": "Greek–Hellenistic",
    "roman": "Greek–Hellenistic",
    "stoic": "Greek–Hellenistic",
    "stoicism": "Greek–Hellenistic",
    "hindu": "Indian",
    "buddhist": "Indian",
    "vedic": "Indian",
    "vedanta": "Indian",
    "european": "Modern European",
    "western": "Modern European",
    "enlightenment": "Modern European",
    "zen": "Japanese",
    "shinto": "Japanese",
    "confucian": "Chinese",
    "taoist": "Chinese",
    "daoist": "Chinese",
    "catholic": "Christian",
    "protestant": "Christian",
    "sufi": "Islamic",
    "sunni": "Islamic",
    "kabbalistic": "Jewish",
    "rabbinic": "Jewish",
}


@dataclass
class Scope:
    """
    Represents scope constraints extracted from a query.
    
    Used to filter chunks during retrieval to only include
    sources matching the specified authors, works, traditions, or domains.
    """
    authors: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    traditions: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    strict: bool = True  # If True, filter out non-matching chunks
    
    def is_empty(self) -> bool:
        """Check if scope has any constraints."""
        return not (self.authors or self.titles or self.traditions or self.domains)
    
    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "authors": self.authors,
            "titles": self.titles,
            "traditions": self.traditions,
            "domains": self.domains,
            "strict": self.strict,
        }
    
    def describe(self) -> str:
        """Human-readable description of scope."""
        parts = []
        if self.authors:
            parts.append(f"authors: {', '.join(self.authors)}")
        if self.titles:
            parts.append(f"works: {', '.join(self.titles)}")
        if self.traditions:
            parts.append(f"traditions: {', '.join(self.traditions)}")
        if self.domains:
            parts.append(f"domains: {', '.join(self.domains)}")
        if not parts:
            return "no scope (global)"
        return "; ".join(parts)


def extract_scope(query: str) -> Scope:
    """
    Extract scope constraints from a natural language query.
    
    Patterns recognized:
    - "in Aristotle" / "according to Aristotle" / "Aristotle's view"
    - "in De Anima" / "in the Republic"
    - "in Greek philosophy" / "in the Buddhist tradition"
    - "about ethics" / "on metaphysics"
    
    Args:
        query: Natural language question
    
    Returns:
        Scope object with extracted constraints
    """
    query_lower = query.lower()
    scope = Scope()
    
    # Extract authors
    # Patterns: "in X", "according to X", "X's", "for X", "by X"
    author_patterns = [
        r'\b(?:in|according to|by|from|for)\s+([A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)*)',
        r"([A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)*)'s\s+(?:view|philosophy|thought|theory|account|ethics|metaphysics)",
        r'\bwhat\s+(?:does|did)\s+([A-Z][a-zà-ÿ]+(?:\s+[A-Z][a-zà-ÿ]+)*)\s+(?:say|think|believe|argue|claim)',
    ]
    
    for pattern in author_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            canonical = _resolve_author(match)
            if canonical and canonical not in scope.authors:
                scope.authors.append(canonical)
    
    # Also check for author names directly in query (case-insensitive)
    for alias, canonical in AUTHOR_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            if canonical not in scope.authors:
                scope.authors.append(canonical)
    
    # Extract work titles
    # Patterns: "in the X", "in X", "from X"
    for alias, canonical in WORK_ALIASES.items():
        if re.search(r'\b(?:in\s+(?:the\s+)?|from\s+(?:the\s+)?)' + re.escape(alias) + r'\b', query_lower):
            if canonical not in scope.titles:
                scope.titles.append(canonical)
    
    # Extract traditions
    for alias, canonical in TRADITION_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', query_lower):
            if canonical not in scope.traditions:
                scope.traditions.append(canonical)
    
    # Extract domains (less aggressive - only if explicitly mentioned)
    domain_patterns = [
        r'\b(?:about|on|concerning|regarding)\s+(ethics|metaphysics|epistemology|theology|anthropology|history)\b',
        r'\b(ethical|metaphysical|epistemological|theological|anthropological|historical)\s+(?:question|issue|problem|view)',
    ]
    domain_map = {
        "ethics": "Ethics", "ethical": "Ethics",
        "metaphysics": "Metaphysics", "metaphysical": "Metaphysics",
        "epistemology": "Epistemology", "epistemological": "Epistemology",
        "theology": "Theology", "theological": "Theology",
        "anthropology": "Anthropology", "anthropological": "Anthropology",
        "history": "History", "historical": "History",
    }
    
    for pattern in domain_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            domain = domain_map.get(match.lower())
            if domain and domain not in scope.domains:
                scope.domains.append(domain)
    
    return scope


def _resolve_author(name: str) -> str | None:
    """Resolve author name to canonical form."""
    name_lower = name.lower().strip()
    
    # Direct alias match
    if name_lower in AUTHOR_ALIASES:
        return AUTHOR_ALIASES[name_lower]
    
    # Case-insensitive match against valid authors
    for valid in VALID_AUTHORS:
        if valid.lower() == name_lower:
            return valid
    
    # Partial match (e.g., "Kant" -> "Immanuel Kant")
    for valid in VALID_AUTHORS:
        if name_lower in valid.lower():
            return valid
    
    return None


class ScopeFilter:
    """
    Apply scope constraints to chunk retrieval.
    
    Usage:
        scope = extract_scope(query)
        filter = ScopeFilter(storage, scope)
        
        # Filter chunk IDs
        scoped_ids = filter.filter_chunk_ids(chunk_ids)
        
        # Or get text_ids for SQL queries
        text_ids = filter.get_text_ids()
    """
    
    def __init__(self, storage: "DuckDBStorage", scope: Scope):
        self.storage = storage
        self.scope = scope
        self._text_ids: set[int] | None = None
    
    def get_text_ids(self) -> set[int]:
        """Get text_ids matching the scope constraints."""
        if self._text_ids is not None:
            return self._text_ids
        
        if self.scope.is_empty():
            # No scope = all text_ids
            rows = self.storage.con.execute("SELECT text_id FROM files").fetchall()
            self._text_ids = {r[0] for r in rows}
            return self._text_ids
        
        # Build WHERE clause
        conditions = []
        params = []
        
        if self.scope.authors:
            placeholders = ",".join(["?"] * len(self.scope.authors))
            conditions.append(f"author_source IN ({placeholders})")
            params.extend(self.scope.authors)
        
        if self.scope.titles:
            placeholders = ",".join(["?"] * len(self.scope.titles))
            conditions.append(f"title IN ({placeholders})")
            params.extend(self.scope.titles)
        
        if self.scope.traditions:
            placeholders = ",".join(["?"] * len(self.scope.traditions))
            conditions.append(f"tradition IN ({placeholders})")
            params.extend(self.scope.traditions)
        
        if self.scope.domains:
            # Domains are semicolon-separated, need LIKE matching
            domain_conditions = []
            for domain in self.scope.domains:
                domain_conditions.append("domains LIKE ?")
                params.append(f"%{domain}%")
            conditions.append("(" + " OR ".join(domain_conditions) + ")")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"SELECT text_id FROM files WHERE {where_clause}"
        rows = self.storage.con.execute(sql, params).fetchall()
        self._text_ids = {r[0] for r in rows}
        
        return self._text_ids
    
    def filter_chunk_ids(self, chunk_ids: list[str]) -> list[str]:
        """Filter chunk IDs to only those matching scope."""
        if self.scope.is_empty() or not self.scope.strict:
            return chunk_ids
        
        if not chunk_ids:
            return []
        
        text_ids = self.get_text_ids()
        if not text_ids:
            return []  # No matching texts
        
        # Query chunks that match both chunk_ids and text_ids
        placeholders = ",".join(["?"] * len(chunk_ids))
        text_placeholders = ",".join(["?"] * len(text_ids))
        
        sql = f"""
            SELECT chunk_id FROM chunks
            WHERE chunk_id IN ({placeholders})
            AND text_id IN ({text_placeholders})
        """
        rows = self.storage.con.execute(sql, list(chunk_ids) + list(text_ids)).fetchall()
        
        # Preserve original order
        valid_set = {r[0] for r in rows}
        return [cid for cid in chunk_ids if cid in valid_set]
    
    def get_scoped_chunk_count(self) -> int:
        """Get total number of chunks in scope."""
        text_ids = self.get_text_ids()
        if not text_ids:
            return 0
        
        placeholders = ",".join(["?"] * len(text_ids))
        result = self.storage.con.execute(
            f"SELECT COUNT(*) FROM chunks WHERE text_id IN ({placeholders})",
            list(text_ids)
        ).fetchone()
        return result[0] if result else 0
    
    def get_scoped_chunk_ids(self) -> set[str]:
        """
        Get all chunk_ids that are in scope.
        
        Used by GraphTraverser to filter chunks at collection time.
        This is cached after the first call.
        """
        if not hasattr(self, '_chunk_ids_cache'):
            text_ids = self.get_text_ids()
            if not text_ids:
                self._chunk_ids_cache: set[str] = set()
            else:
                placeholders = ",".join(["?"] * len(text_ids))
                rows = self.storage.con.execute(
                    f"SELECT chunk_id FROM chunks WHERE text_id IN ({placeholders})",
                    list(text_ids)
                ).fetchall()
                self._chunk_ids_cache = {r[0] for r in rows}
        return self._chunk_ids_cache
    
    def get_scoped_edges(self) -> set[tuple[str, str, str]]:
        """
        Get all edges (subject, object, predicate) that have in-scope provenance.
        
        An edge is in-scope if at least one of its supporting chunks is in scope.
        Used by GraphTraverser to constrain traversal to scoped provenance paths.
        
        This is cached after the first call.
        """
        if not hasattr(self, '_edges_cache'):
            chunk_ids = self.get_scoped_chunk_ids()
            if not chunk_ids:
                self._edges_cache: set[tuple[str, str, str]] = set()
            else:
                self._edges_cache = self.storage.get_scoped_edges(chunk_ids)
        return self._edges_cache
    
    def derive_communities(
        self, 
        node_to_community: dict[str, int],
        top_n: int = 5,
    ) -> list[tuple[int, int]]:
        """
        Derive relevant communities from scoped chunks.
        
        Maps: scoped chunks -> entities -> global comm_ids -> ranked by overlap.
        Replaces global community report routing for strict scope queries.
        
        Args:
            node_to_community: Global node->community mapping
            top_n: Number of top communities to return
        
        Returns:
            List of (community_id, entity_count) tuples, sorted by count desc
        """
        chunk_ids = self.get_scoped_chunk_ids()
        return self.storage.derive_scoped_communities(chunk_ids, node_to_community, top_n)
