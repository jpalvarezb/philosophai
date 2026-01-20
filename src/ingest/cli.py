"""CLI for running the ingest pipeline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="PhilosophAI Ingest Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (clean -> canonicalize -> embed -> communities -> reports)
  python -m src.ingest.cli --db data/philosoph.duckdb --all

  # Run specific steps
  python -m src.ingest.cli --db data/philosoph.duckdb --clean
  python -m src.ingest.cli --db data/philosoph.duckdb --canonicalize
  python -m src.ingest.cli --db data/philosoph.duckdb --embed
  python -m src.ingest.cli --db data/philosoph.duckdb --communities
  python -m src.ingest.cli --db data/philosoph.duckdb --reports
        """,
    )
    
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--clean", action="store_true", help="Clean triples (remove noise)")
    parser.add_argument("--canonicalize", action="store_true", help="Canonicalize entities")
    parser.add_argument("--embed", action="store_true", help="Embed chunks")
    parser.add_argument("--communities", action="store_true", help="Detect communities")
    parser.add_argument("--reports", action="store_true", help="Generate community reports")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    
    args = parser.parse_args()
    
    # Check for OpenAI key if needed
    openai_key = os.environ.get("OPENAI_API_KEY")
    needs_openai = args.embed or args.communities or args.reports or args.all
    if needs_openai and not openai_key:
        print("❌ OPENAI_API_KEY environment variable required for embedding/communities/reports")
        return 1
    
    # Import modules
    from ..storage import DuckDBStorage
    from .cleaner import TripleCleaner
    from .canonicalizer import EntityCanonicalizer
    from .embedder import ChunkEmbedder
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1
    
    print(f"📂 Opening database: {db_path}")
    storage = DuckDBStorage(db_path)
    
    try:
        # Step 1: Clean triples
        if args.clean or args.all:
            print("\n" + "=" * 60)
            print("STEP 1: Clean Triples")
            print("=" * 60)
            cleaner = TripleCleaner(storage)
            cleaner.clean(dry_run=args.dry_run)
        
        # Step 2: Canonicalize entities
        if args.canonicalize or args.all:
            print("\n" + "=" * 60)
            print("STEP 2: Canonicalize Entities")
            print("=" * 60)
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                canonicalizer = EntityCanonicalizer(storage)
                canonicalizer.canonicalize()
        
        # Step 3: Embed chunks
        if args.embed or args.all:
            print("\n" + "=" * 60)
            print("STEP 3: Embed Chunks")
            print("=" * 60)
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                embedder = ChunkEmbedder(storage, client)
                embedder.embed_chunks()
        
        # Step 4: Detect communities
        if args.communities or args.all:
            print("\n" + "=" * 60)
            print("STEP 4: Detect Communities")
            print("=" * 60)
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                from openai import OpenAI
                from ..graph import GraphBuilder, CommunityDetector
                
                client = OpenAI(api_key=openai_key)
                builder = GraphBuilder(storage)
                G = builder.build()
                
                detector = CommunityDetector(
                    graph=G,
                    storage=storage,
                    llm_client=client,
                )
                detector.build_and_store_communities(
                    resolution=1.2,
                    min_community_size=5,
                    summarize=True,
                )
                
                # Populate membership table
                storage.populate_community_membership()
        
        # Step 5: Generate community reports
        if args.reports or args.all:
            print("\n" + "=" * 60)
            print("STEP 5: Generate Community Reports")
            print("=" * 60)
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                from openai import OpenAI
                from ..graph import GraphBuilder, CommunityReportGenerator
                
                client = OpenAI(api_key=openai_key)
                builder = GraphBuilder(storage)
                G = builder.build()
                
                # Build node_to_community mapping
                communities_df = storage.get_communities()
                node_to_community = {}
                for _, row in communities_df.iterrows():
                    for node_id in row["node_ids"]:
                        node_to_community[node_id] = row["community_id"]
                
                generator = CommunityReportGenerator(
                    storage=storage,
                    graph=G,
                    node_to_community=node_to_community,
                    llm_client=client,
                )
                generator.generate_all_reports(min_community_size=5)
        
        print("\n" + "=" * 60)
        print("✅ Pipeline complete!")
        print("=" * 60)
        
        # Print summary
        print("\n📊 Database Summary:")
        tables = storage.con.execute("SHOW TABLES").fetchall()
        for t in tables:
            count = storage.con.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
            print(f"   {t[0]}: {count:,} rows")
    
    finally:
        storage.close()
    
    return 0


if __name__ == "__main__":
    exit(main())
