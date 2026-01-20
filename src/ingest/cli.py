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

  # Run resolution sweep to find optimal Leiden resolution
  python -m src.ingest.cli --db data/philosoph.duckdb --resolution-sweep

  # Detect communities with custom resolution and min edge weight
  python -m src.ingest.cli --db data/philosoph.duckdb --communities --resolution 0.6 --min-edge-weight 3
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
    
    # Community detection options
    parser.add_argument("--resolution", type=float, default=0.8, help="Leiden resolution (default: 0.8)")
    parser.add_argument("--min-edge-weight", type=int, default=1, help="Min edge support for clustering (default: 1, keep all edges)")
    parser.add_argument("--resolution-sweep", action="store_true", help="Run resolution sweep comparison (no storage)")
    parser.add_argument("--no-summarize", action="store_true", help="Skip LLM summarization of communities")
    
    # Report generation options
    parser.add_argument("--min-report-size", type=int, default=20, help="Min community size for report generation (default: 20)")
    parser.add_argument("--max-reports", type=int, default=200, help="Max number of community reports to generate (default: 200)")
    
    args = parser.parse_args()
    
    # Check for OpenAI key if needed
    openai_key = os.environ.get("OPENAI_API_KEY")
    needs_openai = args.embed or args.canonicalize or args.communities or args.reports or args.all
    if needs_openai and not openai_key:
        print("❌ OPENAI_API_KEY environment variable required for canonicalize/embed/communities/reports")
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
        
        # Step 2: Canonicalize entities + predicates (with semantic embeddings)
        if args.canonicalize or args.all:
            print("\n" + "=" * 60)
            print("STEP 2: Canonicalize Entities & Predicates")
            print("=" * 60)
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                canonicalizer = EntityCanonicalizer(storage, llm_client=client)
                canonicalizer.canonicalize_all()
        
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
        
        # Build graphs if needed for communities or reports
        G = None  # Full directed graph for traversal/reports
        cluster_G = None  # Cluster projection for Leiden
        node_to_community = None
        
        if (args.communities or args.reports or args.resolution_sweep or args.all) and not args.dry_run:
            from ..graph import GraphBuilder
            builder = GraphBuilder(storage)
            
            # Build cluster projection for community detection
            print("\n" + "=" * 60)
            print("Building Cluster Projection")
            print("=" * 60)
            cluster_G = builder.build_cluster_projection()
            
            # Build full directed graph for reports/traversal
            if args.reports or args.all:
                print("\n" + "=" * 60)
                print("Building Full Graph (for reports)")
                print("=" * 60)
                G = builder.build()
        
        # Resolution sweep (standalone analysis, no storage)
        if args.resolution_sweep:
            print("\n" + "=" * 60)
            print("Resolution Sweep Analysis")
            print("=" * 60)
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                from ..graph import CommunityDetector
                detector = CommunityDetector(
                    storage=storage,
                    cluster_graph=cluster_G,
                    llm_client=None,  # No LLM needed for sweep
                )
                detector.resolution_sweep(
                    min_edge_weight=args.min_edge_weight,
                )
                print("\n💡 Pick a resolution and re-run with --communities --resolution <value>")
        
        # Step 4: Detect communities
        if args.communities or args.all:
            print("\n" + "=" * 60)
            print("STEP 4: Detect Communities")
            print("=" * 60)
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                from openai import OpenAI
                from ..graph import CommunityDetector
                
                client = OpenAI(api_key=openai_key)
                
                detector = CommunityDetector(
                    storage=storage,
                    cluster_graph=cluster_G,
                    graph=G,  # For reporting (may be None)
                    llm_client=client,
                )
                detector.build_and_store_communities(
                    resolution=args.resolution,
                    min_community_size_for_summary=args.min_report_size,
                    min_edge_weight=args.min_edge_weight,
                    summarize=not args.no_summarize,
                )
                
                # Populate membership table (now includes ALL nodes)
                storage.populate_community_membership()
                
                # Store node_to_community for reports step
                node_to_community = detector.node_to_community
        
        # Step 5: Generate community reports
        if args.reports or args.all:
            print("\n" + "=" * 60)
            print("STEP 5: Generate Community Reports")
            print("=" * 60)
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                from openai import OpenAI
                from ..graph import CommunityReportGenerator
                
                client = OpenAI(api_key=openai_key)
                
                # Build node_to_community mapping if not already available
                if node_to_community is None:
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
                generator.generate_all_reports(
                    min_community_size=args.min_report_size,
                    max_communities=args.max_reports,
                )
        
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
