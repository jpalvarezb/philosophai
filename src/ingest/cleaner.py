"""Triple cleaning - filter noise and metadata from extracted triples."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import IngestConfig

if TYPE_CHECKING:
    from ..storage import DuckDBStorage
# Keep only alphabetical tokens and reasonable lengths
ALPHA_FILTER_SQL = """
    -- Require alphabetic (allow spaces/hyphens) for subject/predicate/object
    subject_norm ~ '^[A-Za-z][A-Za-z \\-]*$'
    AND predicate_norm ~ '^[A-Za-z][A-Za-z \\-]*$'
    AND (
        object_norm IS NULL
        OR object_norm = ''
        OR object_norm ~ '^[A-Za-z][A-Za-z \\-]*$'
    )
    -- Require subject/object length > 1 (after trimming); allow empty object for unary facts
    AND LENGTH(TRIM(subject_norm)) > 1
    AND (
        object_norm IS NULL
        OR object_norm = ''
        OR LENGTH(TRIM(object_norm)) > 1
    )
"""


# Filter patterns for Project Gutenberg and other metadata noise
LEGACY_NOISE_FILTER_SQL = """
    -- Project Gutenberg (all variants)
    LOWER(subject_norm) LIKE '%gutenberg%' 
    OR LOWER(object_norm) LIKE '%gutenberg%'
    
    -- eBook/ebook
    OR LOWER(subject_norm) LIKE '%ebook%' 
    OR LOWER(object_norm) LIKE '%ebook%'
    
    -- trademark
    OR LOWER(subject_norm) LIKE '%trademark%' 
    OR LOWER(object_norm) LIKE '%trademark%'
    
    -- copyright
    OR LOWER(subject_norm) LIKE '%copyright%' 
    OR LOWER(object_norm) LIKE '%copyright%'
    
    -- license
    OR LOWER(subject_norm) LIKE '%license%' 
    OR LOWER(object_norm) LIKE '%license%'
    
    -- Foundation (PG-related: Archive, Literary, EIN, tax, etc.)
    OR (
        (LOWER(subject_norm) LIKE '%foundation%' OR LOWER(object_norm) LIKE '%foundation%')
        AND (
            LOWER(subject_norm) LIKE '%archive%'
            OR LOWER(object_norm) LIKE '%archive%'
            OR LOWER(subject_norm) LIKE '%literary%'
            OR LOWER(object_norm) LIKE '%ein%'
            OR LOWER(object_norm) LIKE '%tax%'
            OR LOWER(object_norm) LIKE '%mississippi%'
            OR LOWER(object_norm) LIKE '%501%'
            OR LOWER(object_norm) LIKE '%charity%'
            OR LOWER(predicate_norm) LIKE '%indemnif%'
            OR LOWER(predicate_norm) LIKE '%liable%'
            OR LOWER(object_norm) LIKE '%salt lake%'
            OR LOWER(object_norm) LIKE '%809 north%'
            OR LOWER(subject_norm) = 'foundation'
            OR LOWER(object_norm) = 'foundation'
        )
    )
    
    -- volunteers/donations
    OR LOWER(subject_norm) LIKE '%volunteer%' 
    OR LOWER(object_norm) LIKE '%volunteer%'
    OR LOWER(subject_norm) LIKE '%donation%' 
    OR LOWER(object_norm) LIKE '%donation%'
    
    -- URLs
    OR LOWER(subject_norm) LIKE '%www%' 
    OR LOWER(object_norm) LIKE '%www%'
    OR LOWER(subject_norm) LIKE '%http%' 
    OR LOWER(object_norm) LIKE '%http%'
    
    -- refund/warranty/disclaimer
    OR LOWER(subject_norm) LIKE '%refund%' 
    OR LOWER(object_norm) LIKE '%refund%'
    OR LOWER(subject_norm) LIKE '%warranty%'
    OR LOWER(object_norm) LIKE '%warranty%'
    OR LOWER(subject_norm) LIKE '%disclaimer%'
    OR LOWER(object_norm) LIKE '%disclaimer%'
    
    -- public domain
    OR LOWER(object_norm) LIKE '%public domain%'
    OR LOWER(subject_norm) LIKE '%public domain%'
    
    -- electronic work
    OR LOWER(subject_norm) LIKE '%electronic work%' 
    OR LOWER(object_norm) LIKE '%electronic work%'
    
    -- Michael Hart
    OR LOWER(subject_norm) LIKE '%michael%hart%' 
    OR LOWER(object_norm) LIKE '%michael%hart%'
    
    -- ASCII/plain vanilla
    OR LOWER(subject_norm) LIKE '%ascii%' 
    OR LOWER(object_norm) LIKE '%ascii%'
    OR LOWER(subject_norm) LIKE '%plain vanilla%' 
    OR LOWER(object_norm) LIKE '%plain vanilla%'
    
    -- user (legal indemnify context)
    OR (LOWER(subject_norm) = 'user' AND LOWER(predicate_norm) LIKE '%agree%')
    
    -- Additional high-frequency PG patterns
    OR LOWER(subject_norm) LIKE '%royalty%'
    OR LOWER(object_norm) LIKE '%royalty%'
    OR LOWER(subject_norm) LIKE '%updated editions%'
    OR LOWER(object_norm) LIKE '%updated editions%'
    OR LOWER(predicate_norm) LIKE '%links to%' AND object_norm LIKE '/%'
    OR LOWER(subject_norm) LIKE '%menu item%'
    OR LOWER(subject_norm) LIKE '%subnav%'
    OR LOWER(subject_norm) LIKE '%icon%depicts%'
    OR (LOWER(subject_norm) LIKE '%icon%' AND LOWER(predicate_norm) LIKE '%depicts%')
    OR LOWER(subject_norm) LIKE '%distributor%'
    OR LOWER(object_norm) LIKE '%distributor%'
    OR LOWER(predicate_norm) LIKE '%appears on page%'
    OR LOWER(subject_norm) LIKE '%paragraph%1.e%'
    OR LOWER(object_norm) LIKE '%paragraph%1.e%'
    OR LOWER(subject_norm) LIKE '%redistribution%'
    OR LOWER(object_norm) LIKE '%redistribution%'
    OR LOWER(subject_norm) LIKE '%internet archive%'
    OR LOWER(object_norm) LIKE '%internet archive%'
    OR LOWER(subject_norm) LIKE '%archive-it%'
    OR LOWER(object_norm) LIKE '%archive-it%'
    OR LOWER(subject_norm) LIKE '%librivox%'
    OR LOWER(object_norm) LIKE '%librivox%'
    OR subject_norm LIKE '[%]'
    OR object_norm LIKE '[%]'
    OR subject_norm LIKE '1.e.%'
    OR subject_norm LIKE '1.f.%'
    OR LOWER(object_norm) LIKE '%reasonable fee%'
    OR LOWER(predicate_norm) LIKE '%tax deductible%'
    OR LOWER(object_norm) LIKE '%tax deductible%'
    OR (LOWER(subject_norm) LIKE '%contributions%' AND LOWER(object_norm) LIKE '%u.s.%')
    OR LOWER(subject_norm) LIKE '%invalid provision%'
    OR LOWER(subject_norm) LIKE '%defect in work%'
    OR LOWER(object_norm) LIKE '%within 90 days%'
    OR LOWER(subject_norm) LIKE '%save page%'
    OR LOWER(subject_norm) LIKE '%search icon%'
    OR LOWER(subject_norm) LIKE '%hamburger icon%'
    OR LOWER(subject_norm) LIKE '%web page%' AND LOWER(predicate_norm) LIKE '%used as%'
    OR LOWER(subject_norm) LIKE '%individual works%'
    OR (LOWER(subject_norm) LIKE '%foundation ein%')
    
    -- Chapter/page references
    OR LOWER(subject_norm) LIKE '%chapter%'
    OR LOWER(object_norm) LIKE '%chapter%'
    OR subject_norm ~ '^[0-9]+,[0-9]+$'
    OR subject_norm ~ '^[0-9]+-[0-9]+$'
    OR LOWER(subject_norm) LIKE '%has vignette%'
    OR LOWER(predicate_norm) LIKE '%on pages%'
    OR LOWER(predicate_norm) LIKE '%appears on page%'
    
    -- Survey/demographic fieldwork data
    OR LOWER(subject_norm) LIKE '%pastor school%'
    OR LOWER(subject_norm) LIKE '%test scores%'
    OR LOWER(subject_norm) LIKE '%menstrual%'
    OR LOWER(subject_norm) LIKE '%physical defects%'
    OR LOWER(subject_norm) LIKE '%best friends%'
    OR LOWER(subject_norm) LIKE '%older brothers number%'
    OR LOWER(subject_norm) LIKE '%younger sisters number%'
    OR LOWER(subject_norm) LIKE '%half brother%'
    OR LOWER(subject_norm) LIKE '%half sister%'
    OR LOWER(subject_norm) LIKE '%father remarried%'
    OR LOWER(subject_norm) LIKE '%parents divorced%'
    OR LOWER(subject_norm) LIKE '%mother dead%'
    OR LOWER(subject_norm) LIKE '%father dead%'
    OR LOWER(subject_norm) LIKE '%matrilocal%'
    OR LOWER(subject_norm) LIKE '%patrilocal%'
    OR LOWER(subject_norm) LIKE '%oldest children%'
    OR LOWER(subject_norm) LIKE '%english knowledge%'
    
    -- Bibliographic metadata  
    OR LOWER(subject_norm) LIKE '%reimpression%'
    OR LOWER(subject_norm) LIKE '%grynaeus%'
    OR LOWER(subject_norm) LIKE '%obscurae subtilitatis%'
    
    -- Onomatopoeia/sound descriptions
    OR LOWER(subject_norm) LIKE '%tudududu%'
    OR LOWER(subject_norm) LIKE '%flute sounds%'
    OR LOWER(subject_norm) LIKE '%pipe sounds%'
    OR LOWER(subject_norm) LIKE '%trumpet sounds%'
    OR LOWER(subject_norm) LIKE '%wheel creaking%'
    OR LOWER(subject_norm) LIKE '%hail noise%'
    OR LOWER(subject_norm) LIKE '%wind noise%'
    OR (LOWER(subject_norm) LIKE '%dog bark%' AND LOWER(predicate_norm) NOT LIKE '%symbol%')
    OR (LOWER(subject_norm) LIKE '%sheep bleat%' AND LOWER(predicate_norm) NOT LIKE '%symbol%')
"""


class TripleCleaner:
    """Clean extracted triples by removing noise and metadata."""

    def __init__(self, storage: "DuckDBStorage", config: IngestConfig | None = None):
        self.storage = storage
        self.config = config or IngestConfig()

    @staticmethod
    def _escape_like(pattern: str) -> str:
        return pattern.replace("'", "''")

    def _build_noise_filter(self) -> str:
        """Build the active noise SQL from dynamic rules or fallback to legacy SQL."""
        entity_patterns = list(self.config.noise_entity_patterns)
        predicate_patterns = list(self.config.noise_predicate_patterns)
        if self.config.cleaning_rules:
            entity_patterns.extend(self.config.cleaning_rules.entity_patterns)
            predicate_patterns.extend(self.config.cleaning_rules.predicate_patterns)

        clauses: list[str] = []
        for pattern in sorted(set(entity_patterns)):
            escaped = self._escape_like(pattern.lower())
            clauses.append(f"LOWER(subject_norm) LIKE '{escaped}'")
            clauses.append(f"LOWER(object_norm) LIKE '{escaped}'")

        for pattern in sorted(set(predicate_patterns)):
            escaped = self._escape_like(pattern.lower())
            clauses.append(f"LOWER(predicate_norm) LIKE '{escaped}'")

        provenance_failed = []
        if self.config.cleaning_rules:
            provenance_failed = self.config.cleaning_rules.provenance_failed_entities
        for entity in sorted(set(provenance_failed)):
            escaped = self._escape_like(entity.lower())
            clauses.append(f"LOWER(subject_norm) = '{escaped}'")
            clauses.append(f"LOWER(object_norm) = '{escaped}'")

        type_rules = self.config.cleaning_rules.type_rules if self.config.cleaning_rules else {}
        for entity_type, patterns in type_rules.items():
            escaped_type = self._escape_like(entity_type)
            for pattern in patterns:
                escaped_pattern = self._escape_like(pattern.lower())
                clauses.append(
                    f"(subject_type = '{escaped_type}' AND LOWER(subject_norm) LIKE '{escaped_pattern}')"
                )
                clauses.append(
                    f"(object_type = '{escaped_type}' AND LOWER(object_norm) LIKE '{escaped_pattern}')"
                )

        if not clauses:
            return LEGACY_NOISE_FILTER_SQL
        return "\n    OR ".join(clauses)

    def clean(
        self,
        source_table: str = "normalized_triples",
        target_table: str = "normalized_triples_clean",
        dry_run: bool = False,
    ) -> dict:
        """
        Clean triples by filtering noise patterns.
        
        Args:
            source_table: Table with normalized triples
            target_table: Table to create with cleaned triples
            dry_run: If True, only report what would be removed
        
        Returns:
            Dict with statistics
        """
        con = self.storage.con
        
        # Count before
        total = con.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()[0]
        print(f"📊 Total triples in {source_table}: {total:,}")

        noise_filter_sql = self._build_noise_filter()

        # Count noise
        noise_count = con.execute(
            f"SELECT COUNT(*) FROM {source_table} WHERE {noise_filter_sql}"
        ).fetchone()[0]
        print(f"🗑️  Noise triples: {noise_count:,} ({100*noise_count/total:.1f}%)")
        
        if dry_run:
            # Preview what would be removed
            print("\n📋 Sample triples to be removed:")
            sample = con.execute(f"""
                SELECT subject_norm, predicate_norm, object_norm 
                FROM {source_table} 
                WHERE {noise_filter_sql}
                LIMIT 10
            """).fetchdf()
            for _, row in sample.iterrows():
                print(f"   ({row['subject_norm']}, {row['predicate_norm']}, {row['object_norm']})")
            
            return {
                "total": total,
                "noise": noise_count,
                "clean": total - noise_count,
                "dry_run": True,
            }
        
        # Create cleaned table
        print(f"\n🔧 Creating {target_table}...")
        con.execute(f"DROP TABLE IF EXISTS {target_table}")
        con.execute(f"""
            CREATE TABLE {target_table} AS
            SELECT * FROM {source_table}
            WHERE NOT ({noise_filter_sql})
              AND ({ALPHA_FILTER_SQL})
        """)
        
        # Count after
        clean_count = con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
        print(f"✅ Clean triples: {clean_count:,}")
        
        return {
            "total": total,
            "noise": noise_count,
            "clean": clean_count,
            "dry_run": False,
        }

    def add_custom_filter(self, pattern: str):
        """Add a custom filter pattern (for corpus-specific noise)."""
        self.config.noise_entity_patterns.append(pattern)
