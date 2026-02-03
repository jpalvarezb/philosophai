"""Triple cleaning - filter noise and metadata from extracted triples."""
from __future__ import annotations

from typing import TYPE_CHECKING

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
NOISE_FILTER_SQL = """
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

    def __init__(self, storage: "DuckDBStorage"):
        self.storage = storage

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
        
        # Count noise
        noise_count = con.execute(
            f"SELECT COUNT(*) FROM {source_table} WHERE {NOISE_FILTER_SQL}"
        ).fetchone()[0]
        print(f"🗑️  Noise triples: {noise_count:,} ({100*noise_count/total:.1f}%)")
        
        if dry_run:
            # Preview what would be removed
            print("\n📋 Sample triples to be removed:")
            sample = con.execute(f"""
                SELECT subject_norm, predicate_norm, object_norm 
                FROM {source_table} 
                WHERE {NOISE_FILTER_SQL}
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
            WITH filtered AS (
                SELECT * FROM {source_table}
                WHERE NOT ({NOISE_FILTER_SQL})
                  AND ({ALPHA_FILTER_SQL})
            ),
            pred_support AS (
                SELECT
                    subject_norm,
                    object_norm,
                    predicate_norm,
                    COUNT(*) AS support
                FROM filtered
                WHERE object_norm IS NOT NULL AND object_norm != ''
                GROUP BY 1, 2, 3
            ),
            ranked AS (
                SELECT
                    f.*,
                    ps.support,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.subject_norm, f.object_norm
                        ORDER BY ps.support DESC NULLS LAST, f.predicate_norm
                    ) AS pred_rank
                FROM filtered f
                LEFT JOIN pred_support ps
                  ON f.subject_norm = ps.subject_norm
                 AND f.object_norm = ps.object_norm
                 AND f.predicate_norm = ps.predicate_norm
            )
            SELECT * EXCLUDE(pred_rank, support)
            FROM ranked
            WHERE (object_norm IS NULL OR object_norm = '' OR pred_rank = 1)
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
        global NOISE_FILTER_SQL
        NOISE_FILTER_SQL = f"{NOISE_FILTER_SQL}\n    OR {pattern}"
