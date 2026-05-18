"""CLI for running the ingestion pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..config import setup_logging
from ..storage import DuckDBStorage
from .agents import CleaningAgent, CorpusAuditAgent
from .canonicalizer import EntityCanonicalizer
from .chunker import CorpusChunker
from .cleaner import TripleCleaner
from .config import load_ingest_config
from .embedder import ChunkEmbedder
from .extractor import TripleExtractor
from .loader import CorpusLoader


def _step(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _build_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for this step"
        )
    return OpenAI(api_key=api_key)


def main():
    parser = argparse.ArgumentParser(
        description="PhilosophAI Ingest Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ingest.cli --db data/philosoph.duckdb --all
  python -m src.ingest.cli --db data/philosoph.duckdb --load --audit --save-config data/ingest_config.json
  python -m src.ingest.cli --db data/philosoph.duckdb --chunk --extract --clean
  python -m src.ingest.cli --db data/philosoph.duckdb --canonicalize --embed --communities --reports
        """,
    )

    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--config", help="Optional config JSON to load")
    parser.add_argument(
        "--save-config", help="Optional config JSON path to persist the active config"
    )

    parser.add_argument(
        "--all", action="store_true", help="Run the full 8-step ingestion pipeline"
    )
    parser.add_argument(
        "--load", action="store_true", help="Load raw files into DuckDB"
    )
    parser.add_argument("--audit", action="store_true", help="Run corpus audit agent")
    parser.add_argument("--chunk", action="store_true", help="Chunk raw texts")
    parser.add_argument("--extract", action="store_true", help="Extract typed triples")
    parser.add_argument(
        "--clean", action="store_true", help="Run dynamic cleaning agent + cleaner"
    )
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Canonicalize entities and predicates",
    )
    parser.add_argument(
        "--embed", action="store_true", help="Embed chunks and entities"
    )
    parser.add_argument("--communities", action="store_true", help="Detect communities")
    parser.add_argument(
        "--reports", action="store_true", help="Generate community reports"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without making changes"
    )

    parser.add_argument(
        "--ocr", action="store_true", help="Enable Tesseract OCR for scanned/image PDFs"
    )
    parser.add_argument("--extraction-model", help="Override extraction model name")
    parser.add_argument("--embedding-model", help="Override embedding model name")
    parser.add_argument(
        "--chunk-method",
        choices=["paragraph", "semantic"],
        help="Override chunking method",
    )
    parser.add_argument(
        "--extract-limit", type=int, help="Optional extraction chunk limit"
    )

    parser.add_argument(
        "--resolution", type=float, default=None, help="Leiden resolution override"
    )
    parser.add_argument(
        "--min-edge-weight",
        type=int,
        default=None,
        help="Min edge support for clustering",
    )
    parser.add_argument(
        "--resolution-sweep",
        action="store_true",
        help="Run resolution sweep comparison",
    )
    parser.add_argument(
        "--no-summarize", action="store_true", help="Skip LLM community summaries"
    )
    parser.add_argument(
        "--min-report-size", type=int, default=20, help="Min community size for reports"
    )
    parser.add_argument(
        "--max-reports", type=int, default=200, help="Max community reports"
    )

    args = parser.parse_args()
    setup_logging()

    config = load_ingest_config(args.config)
    config = config.merge_overrides(
        ocr_enabled=args.ocr or None,
        extraction_model=args.extraction_model,
        embedding_model=args.embedding_model,
        chunk_method=args.chunk_method,
        resolution=args.resolution,
        min_edge_weight=args.min_edge_weight,
    )

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"📂 Opening database: {db_path}")

    run_load = args.load or args.all
    run_audit = args.audit or args.all
    run_chunk = args.chunk or args.all
    run_extract = args.extract or args.all
    run_clean = args.clean or args.all
    run_canonicalize = args.canonicalize or args.all
    run_embed = args.embed or args.all
    run_communities = args.communities or args.all
    run_reports = args.reports or args.all

    storage = DuckDBStorage(db_path)
    try:
        if run_load:
            _step("STEP 0: Load Raw Resources")
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                loader = CorpusLoader(storage, config=config)
                print(loader.load_resources())

        llm_client = None
        needs_openai = (
            run_audit
            or run_clean
            or run_canonicalize
            or run_embed
            or run_communities
            or run_reports
        )
        if needs_openai and not args.dry_run:
            llm_client = _build_openai_client()

        if run_audit:
            _step("STEP 1: Corpus Audit")
            agent = CorpusAuditAgent(storage, llm_client=llm_client, config=config)
            config, report = agent.run()
            print(
                {
                    "chunk_recommendations": report.chunk_recommendations,
                    "boilerplate_patterns": len(report.boilerplate_patterns),
                    "discovered_subtypes": report.discovered_subtypes,
                }
            )

        if run_chunk:
            _step("STEP 2: Chunk Documents")
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                chunker = CorpusChunker(storage, config=config)
                print(chunker.chunk_all())

        if run_extract:
            _step("STEP 3: Extract Typed Triples")
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                extractor = TripleExtractor(storage, config=config)
                print(extractor.extract_pending(limit=args.extract_limit))

        if run_clean:
            _step("STEP 4: Dynamic Clean")
            cleaning_agent = CleaningAgent(
                storage, llm_client=llm_client, config=config
            )
            config, rules = cleaning_agent.run()
            print(
                {
                    "entity_patterns": rules.entity_patterns,
                    "predicate_patterns": rules.predicate_patterns,
                    "provenance_failed_entities": len(rules.provenance_failed_entities),
                }
            )
            if args.dry_run:
                print("⏭️  Skipping cleaner write in dry-run mode")
            else:
                cleaner = TripleCleaner(storage, config=config)
                cleaner.clean(dry_run=False)

        if run_canonicalize:
            _step("STEP 5: Canonicalize Entities & Predicates")
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                canonicalizer = EntityCanonicalizer(
                    storage, llm_client=llm_client, config=config
                )
                canonicalizer.canonicalize_all()

        if run_embed:
            _step("STEP 6: Embed Chunks & Entities")
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                embedder = ChunkEmbedder(storage, llm_client=llm_client, config=config)
                print(embedder.embed_chunks())
                print(embedder.embed_entities())

        G = None
        cluster_G = None
        node_to_community = None
        if (
            run_communities or run_reports or args.resolution_sweep
        ) and not args.dry_run:
            from ..graph import GraphBuilder

            builder = GraphBuilder(storage)
            _step("Building Cluster Projection")
            cluster_G = builder.build_cluster_projection()
            if run_reports:
                _step("Building Full Graph")
                G = builder.build()

        if args.resolution_sweep:
            _step("Resolution Sweep Analysis")
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                from ..graph import CommunityDetector

                detector = CommunityDetector(
                    storage=storage,
                    cluster_graph=cluster_G,
                    llm_client=None,
                )
                detector.resolution_sweep(min_edge_weight=config.min_edge_weight)

        if run_communities:
            _step("STEP 7: Detect Communities")
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                from ..graph import CommunityDetector

                detector = CommunityDetector(
                    storage=storage,
                    cluster_graph=cluster_G,
                    graph=G,
                    llm_client=llm_client,
                )
                detector.build_and_store_communities(
                    resolution=config.resolution,
                    min_community_size_for_summary=args.min_report_size,
                    min_edge_weight=config.min_edge_weight,
                    summarize=not args.no_summarize,
                )
                storage.populate_community_membership()
                node_to_community = detector.node_to_community

        if run_reports:
            _step("STEP 8: Generate Community Reports")
            if args.dry_run:
                print("⏭️  Skipping in dry-run mode")
            else:
                from ..graph import CommunityReportGenerator

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
                    llm_client=llm_client,
                )
                generator.generate_all_reports(
                    min_community_size=args.min_report_size,
                    max_communities=args.max_reports,
                )

        if args.save_config:
            saved_path = config.save(args.save_config)
            print(f"💾 Saved config to {saved_path}")

        print("\n" + "=" * 60)
        print("✅ Pipeline complete!")
        print("=" * 60)

        print("\n📊 Database Summary:")
        for (table_name,) in storage.con.execute("SHOW TABLES").fetchall():
            count = storage.con.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]
            print(f"   {table_name}: {count:,} rows")
    finally:
        storage.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
