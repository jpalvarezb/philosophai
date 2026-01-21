"""Multi-hop reasoning agent with community-routed GraphRAG."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from ..config import scope_logger

if TYPE_CHECKING:
    from openai import OpenAI
    from ..storage import DuckDBStorage
    from ..graph import GraphBuilder, GraphTraverser, GraphFilters
    from ..rag import VectorSearch, ResultFusion, CitationBuilder
    from .tools import AgentTools
    from .trace import TraceRecorder


class MultiHopAgent:
    """
    Orchestrates multi-hop reasoning over the knowledge graph with GraphRAG routing.

    Pipeline:
    1. Search community reports → select target communities
    2. Restrict entity resolution to target communities
    3. Community-gated graph traversal
    4. Pre-filter chunks to community-linked evidence
    5. Fuse results and select context
    6. Generate answer with citations
    """

    def __init__(
        self,
        storage: "DuckDBStorage",
        graph_builder: "GraphBuilder",
        vector_search: "VectorSearch",
        fusion: "ResultFusion",
        citation_builder: "CitationBuilder",
        traverser: "GraphTraverser",
        llm_client: "OpenAI",
        node_to_community: dict[str, int],
        filters: "GraphFilters",
        llm_model: str = "gpt-5.2",
        on_event: Callable[[dict], None] | None = None,  # For streaming updates
    ):
        self.storage = storage
        self.graph_builder = graph_builder
        self.vector_search = vector_search
        self.fusion = fusion
        self.citation_builder = citation_builder
        self.traverser = traverser
        self.llm_client = llm_client
        self.node_to_community = node_to_community
        self.filters = filters
        self.llm_model = llm_model
        self.on_event = on_event or (lambda x: None)

    def _emit(self, event_type: str, data: dict):
        """Emit event for streaming UI updates."""
        self.on_event({"type": event_type, **data})

    def query(
        self,
        question: str,
        max_hops: int = 2,
        max_context_chunks: int = 12,
        use_community_routing: bool = True,
        scope: "Scope | None" = None,
    ) -> dict:
        """
        Execute a multi-hop query with community-routed GraphRAG.

        Args:
            question: User's question
            max_hops: Maximum traversal depth
            max_context_chunks: Max chunks to send to LLM
            use_community_routing: If True, use community reports for routing
            scope: Optional Scope object for filtering (from agent's set_scope tool call)

        Returns:
            Dict with answer, citations, and traversal trace
        """
        from .trace import TraceRecorder
        from .scope import ScopeFilter, Scope, ScopeViolationError

        trace = TraceRecorder(query=question)
        
        # Step 0: Apply scope if provided by agent
        scope_filter = ScopeFilter(self.storage, scope) if scope and not scope.is_empty() else None
        
        # Leakage tracking for strict scope enforcement
        scope_debug = {
            "enabled": scope_filter is not None,
            "routing": {"total": 0, "out_of_scope": 0},
            "vector": {"total": 0, "out_of_scope": 0},
            "traversal": {"total": 0, "out_of_scope": 0},
            "context": {"total": 0, "out_of_scope": 0},
        }
        
        # Pre-compute scope data for efficient retrieval-time filtering
        scoped_text_ids: set[int] | None = None
        scoped_chunks: set[str] | None = None
        scoped_edges: set[tuple[str, str, str]] | None = None
        is_strict_scope = scope_filter is not None and scope.strict
        
        if scope_filter:
            scoped_text_ids = scope_filter.get_text_ids()
            scoped_chunks = scope_filter.get_scoped_chunk_ids()
            scoped_chunk_count = len(scoped_chunks)
            
            # For strict scope: compute scoped edges
            # Entity scope is derived as V(scoped_edges) - not a separate authority
            if is_strict_scope:
                scoped_edges = scope_filter.get_scoped_edges()
                # Derive V(E) for logging only
                scoped_entity_ids = {s for s, _, _ in scoped_edges} | {o for _, o, _ in scoped_edges}
                
                scope_logger.scope_init(
                    scope.describe(),
                    strict=True,
                    texts=len(scoped_text_ids),
                    chunks=scoped_chunk_count,
                    edges=len(scoped_edges),
                    entities=len(scoped_entity_ids)
                )
                
                trace.add_thought(
                    f"Strict scope applied: {scope.describe()}",
                    action="scope_filter",
                    observation=f"Restricting to {scoped_chunk_count:,} chunks, {len(scoped_edges):,} edges, {len(scoped_entity_ids):,} entities",
                )
            else:
                scope_logger.scope_init(
                    scope.describe(),
                    strict=False,
                    texts=len(scoped_text_ids),
                    chunks=scoped_chunk_count
                )
                trace.add_thought(
                    f"Scope applied (non-strict): {scope.describe()}",
                    action="scope_filter",
                    observation=f"Restricting to {scoped_chunk_count:,} chunks from {len(scoped_text_ids)} texts",
                )
            self._emit("status", {"message": f"Scoped to {scope.describe()}..."})

        # Step 1: Community routing
        # For STRICT scope: derive communities from scoped chunks (not global reports)
        # For non-strict scope or no scope: use global community report search
        self._emit("status", {"message": "Routing through communities..."})
        
        target_communities = []
        community_cited_chunks = []
        community_scores = {}
        report_result = None
        
        if is_strict_scope:
            # STRICT SCOPE: Derive communities from scoped chunks -> entities -> global comm_ids
            # This ensures we don't leak through global community reports
            derived_communities = scope_filter.derive_communities(
                self.node_to_community, top_n=5
            )
            target_communities = [comm_id for comm_id, _ in derived_communities]
            community_scores = {comm_id: count / 100.0 for comm_id, count in derived_communities}  # Normalized pseudo-score
            
            # No cited chunks from community reports in strict mode
            # All chunks come from scoped vector search
            community_cited_chunks = []
            
            scope_debug["routing"]["total"] = 0
            scope_debug["routing"]["out_of_scope"] = 0
            scope_debug["routing"]["mode"] = "derived_from_scope"
            
            trace.add_thought(
                f"Strict scope: derived {len(target_communities)} communities from scoped entities",
                action="derive_scoped_communities",
                observation=f"Communities by entity overlap: {derived_communities[:5]}",
            )
            trace.selected_community_ids = target_communities
            trace.community_report_scores = community_scores
            
        elif use_community_routing:
            # NON-STRICT or NO SCOPE: Use global community report search
            try:
                # Pass text_ids for retrieval-time scope filtering of cited chunks
                report_result = self.vector_search.search_community_reports(
                    question, limit=5, text_ids=scoped_text_ids
                )
                target_communities = report_result.community_ids
                community_cited_chunks = report_result.cited_chunk_ids  # Already scoped at SQL level
                community_scores = dict(zip(report_result.community_ids, report_result.scores))
                
                # Track routing - with retrieval-time filtering, out_of_scope should be 0
                if scope_filter:
                    scope_debug["routing"]["total"] = len(community_cited_chunks)
                    scope_debug["routing"]["out_of_scope"] = 0  # All chunks returned are in-scope
                    scope_debug["routing"]["mode"] = "global_reports_filtered"
                
                trace.add_thought(
                    f"Community routing: selected {len(target_communities)} communities",
                    action="community_report_search",
                    action_input={"query": question},
                    observation=f"Communities: {target_communities}, Scores: {[f'{s:.3f}' for s in report_result.scores]}",
                )
                trace.selected_community_ids = target_communities
                trace.community_report_scores = community_scores
            except Exception as e:
                trace.add_thought(f"Community routing failed, falling back to vector search: {e}")
        
        # Step 2: Vector search (scoped at SQL level)
        self._emit("status", {"message": "Searching knowledge base..."})
        
        # Pass text_ids for retrieval-time scope filtering
        vector_result = self.vector_search.search(
            question, chunk_limit=15, community_limit=5, text_ids=scoped_text_ids
        )
        
        # Track vector - with retrieval-time filtering, out_of_scope should be 0
        if scope_filter:
            scope_debug["vector"]["total"] = len(vector_result.chunk_ids)
            scope_debug["vector"]["out_of_scope"] = 0  # All chunks returned are in-scope
        
        # HARD CHECK 1: Vector search must return only scoped chunks
        if is_strict_scope:
            vector_leakage = set(vector_result.chunk_ids) - scoped_chunks
            scope_debug["vector"]["out_of_scope"] = len(vector_leakage)
            
            if vector_leakage:
                scope_logger.check_fail(
                    "vector", 
                    total=len(vector_result.chunk_ids), 
                    out_of_scope=len(vector_leakage), 
                    examples=list(vector_leakage)
                )
                raise ScopeViolationError(
                    stage="vector",
                    message=f"{len(vector_leakage)} out-of-scope chunks returned",
                    examples=list(vector_leakage)[:3],
                )
            else:
                scope_logger.check_pass(
                    "vector", 
                    total=len(vector_result.chunk_ids), 
                    in_scope=len(vector_result.chunk_ids)
                )
        
        # Merge community-cited chunks with vector search results
        # Both are already scoped, no post-filtering needed
        all_chunk_ids = list(dict.fromkeys(community_cited_chunks + vector_result.chunk_ids))
        
        trace.add_thought(
            f"Found {len(vector_result.chunk_ids)} chunks via vector search, {len(community_cited_chunks)} from community reports",
            action="vector_search",
            observation=f"Total unique in-scope chunks: {len(all_chunk_ids)}",
        )

        # Step 3: Query-aware seed selection
        self._emit("status", {"message": "Selecting seed entities..."})
        
        from ..rag import select_seeds
        
        scored_seeds = select_seeds(
            query=question,
            chunk_ids=all_chunk_ids[:30],
            storage=self.storage,
            graph=self.graph_builder.graph,
            node_to_community=self.node_to_community,
            target_communities=target_communities,
            filters=self.filters,
            max_seeds=20,
        )
        
        seed_nodes = [s.entity_id for s in scored_seeds]
        seed_labels = [f"{s.label}({s.score:.2f})" for s in scored_seeds[:5]]
        
        # HARD CHECK 2: Seeds must be in V(scoped_edges)
        if is_strict_scope:
            scoped_entity_ids = {s for s, _, _ in scoped_edges} | {o for _, o, _ in scoped_edges}
            seed_leakage = set(seed_nodes) - scoped_entity_ids
            scope_debug["seeds"] = {"total": len(seed_nodes), "out_of_scope": len(seed_leakage)}
            
            if seed_leakage:
                scope_logger.check_fail(
                    "seed", 
                    total=len(seed_nodes), 
                    out_of_scope=len(seed_leakage), 
                    examples=list(seed_leakage)
                )
                raise ScopeViolationError(
                    stage="seed_selection",
                    message=f"{len(seed_leakage)} out-of-scope seeds selected",
                    examples=list(seed_leakage)[:3],
                )
            else:
                scope_logger.check_pass(
                    "seed", 
                    total=len(seed_nodes), 
                    in_scope=len(seed_nodes)
                )
        
        trace.add_thought(
            f"Selected {len(seed_nodes)} query-relevant seeds",
            action="seed_selection",
            observation=f"Top seeds: {seed_labels}",
        )
        trace.seed_entities = seed_nodes

        # Step 4: Community-gated graph traversal
        self._emit("status", {"message": "Traversing knowledge graph..."})
        trace.add_thought(
            f"Beginning traversal from {len(seed_nodes)} seeds, restricted to {len(target_communities)} communities"
        )

        # Use target communities for gating (stronger restriction)
        traversal_communities = target_communities if target_communities else vector_result.community_ids
        
        # Configure traversal based on scope mode
        # STRICT SCOPE: Use scoped_edges to constrain traversal to in-scope provenance paths
        # NON-STRICT: Only filter chunks at collection time
        traversal_trace = self.traverser.traverse(
            seed_nodes=seed_nodes,
            target_communities=traversal_communities,
            query=question,
            max_hops=max_hops,
            max_nodes=50,
            restrict_to_communities=bool(target_communities) and not is_strict_scope,  # Use community gating only if not strict scope
            scoped_chunks=scoped_chunks,  # Filter chunks at collection time
            scoped_edges=scoped_edges if is_strict_scope else None,  # STRICT: constrain edge traversal
        )
        traversal_trace.seed_chunks = all_chunk_ids[:20]
        
        # HARD CHECK 3: Traversal must only collect scoped chunks
        if is_strict_scope:
            traversal_leakage = set(traversal_trace.collected_chunk_ids) - scoped_chunks
            scope_debug["traversal"]["out_of_scope"] = len(traversal_leakage)
            
            scope_logger.traversal_summary(
                nodes_visited=len(traversal_trace.visited_nodes),
                chunks_collected=len(traversal_trace.collected_chunk_ids),
                edges_filtered=traversal_trace.filtered_out_of_scope_edges
            )
            
            if traversal_leakage:
                scope_logger.check_fail(
                    "traversal", 
                    total=len(traversal_trace.collected_chunk_ids), 
                    out_of_scope=len(traversal_leakage), 
                    examples=list(traversal_leakage)
                )
                raise ScopeViolationError(
                    stage="traversal",
                    message=f"{len(traversal_leakage)} out-of-scope chunks collected",
                    examples=list(traversal_leakage)[:3],
                )
            else:
                scope_logger.check_pass(
                    "traversal", 
                    total=len(traversal_trace.collected_chunk_ids), 
                    in_scope=len(traversal_trace.collected_chunk_ids),
                    extra={"edges_filtered": traversal_trace.filtered_out_of_scope_edges}
                )
        
        # Track traversal
        if scope_filter:
            scope_debug["traversal"]["total"] = len(traversal_trace.collected_chunk_ids)
            scope_debug["traversal"]["out_of_scope"] = 0  # All chunks collected are in-scope
            scope_debug["traversal"]["edges_filtered"] = traversal_trace.filtered_out_of_scope_edges
            scope_debug["traversal"]["mode"] = "strict_edge_constrained" if is_strict_scope else "chunk_filtered"

        trace.add_thought(
            f"Traversal complete: visited {len(traversal_trace.visited_nodes)} nodes, collected {len(traversal_trace.collected_chunk_ids)} in-scope chunks" +
            (f" (filtered {traversal_trace.filtered_out_of_scope_edges} out-of-scope edges)" if is_strict_scope else ""),
            action="graph_traverse",
            observation=f"Communities touched: {list(traversal_trace.visited_communities)}",
        )
        trace.communities_explored = list(traversal_trace.visited_communities)
        trace.nodes_visited = list(traversal_trace.visited_nodes)
        
        # Vector chunks are already scoped at retrieval time
        scoped_vector_chunks = vector_result.chunk_ids

        # Step 5: Select context (with must-keep guarantees)
        self._emit("status", {"message": "Selecting context..."})
        
        # Pass filtered report chunks (not raw report_result which has unfiltered chunks)
        scoped_report_chunks = community_cited_chunks if scope_filter else (
            report_result.cited_chunk_ids if report_result else []
        )
        
        context_chunks, all_collected_ids, context_scores = self.fusion.get_context_with_selection(
            query=question,
            report_cited_chunks=scoped_report_chunks,
            vector_chunks=scoped_vector_chunks,
            trace=traversal_trace,
            max_context=max_context_chunks,
        )

        trace.total_chunks_retrieved = len(all_collected_ids)
        trace.add_thought(
            f"Collected {len(all_collected_ids)} chunks, selected {len(context_chunks)} for LLM context",
            observation=f"Context sources: report={len(community_cited_chunks)}, traversal={len(traversal_trace.collected_chunk_ids)}, vector={len(vector_result.chunk_ids)}",
        )

        # Step 6: Build citations (from context chunks only)
        chunk_ids_for_citations = [c[0] for c in context_chunks]
        
        # Track context leakage (final check)
        if scope_filter:
            scope_debug["context"]["total"] = len(chunk_ids_for_citations)
            context_in_scope = scope_filter.filter_chunk_ids(chunk_ids_for_citations)
            scope_debug["context"]["out_of_scope"] = len(chunk_ids_for_citations) - len(context_in_scope)
        
        citations = self.citation_builder.build_citations(chunk_ids_for_citations)
        trace.final_cited_chunk_ids = chunk_ids_for_citations
        trace.all_collected_chunk_ids = all_collected_ids  # Keep full set for UI

        # Step 7: Generate answer
        self._emit("status", {"message": "Generating answer..."})
        trace.add_thought("Generating answer with citations")

        context_text = self.citation_builder.format_context_for_llm(context_chunks)
        answer = self._generate_answer(question, context_text)

        # Extract citation indices used in answer
        import re
        citation_refs = re.findall(r'\[(\d+)\]', answer)
        citations_used = list(set(int(c) for c in citation_refs if int(c) <= len(citations)))

        trace.set_answer(answer, citations_used)

        self._emit("complete", {"message": "Done"})

        return {
            "answer": answer,
            "citations": self.citation_builder.to_dict_list(citations),
            "traversal": traversal_trace.to_dict(),
            "trace": trace.to_dict(),
            "scope": scope.to_dict() if scope else None,
            "scope_debug": scope_debug if scope_filter else None,
        }

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with context."""
        system_prompt = """You are a knowledgeable assistant answering questions based on provided context.

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Use inline citations like [1], [2] to reference sources
3. If the context doesn't contain enough information, say so
4. Be concise but thorough"""

        user_prompt = f"""Question: {question}

Context:
{context}

Answer the question using the context above. Cite sources with [n] notation."""

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def query_simple(self, question: str, limit: int = 10) -> dict:
        """
        Simple query without graph traversal (fallback mode).
        Uses only vector search on chunks.
        """
        chunk_ids, scores, _ = self.vector_search.search_chunks_only(question, limit=limit)
        chunks = self.storage.get_chunk_texts(chunk_ids)

        citations = self.citation_builder.build_citations(chunk_ids)
        context_text = self.citation_builder.format_context_for_llm(chunks)
        answer = self._generate_answer(question, context_text)

        return {
            "answer": answer,
            "citations": self.citation_builder.to_dict_list(citations),
            "mode": "simple",
        }
