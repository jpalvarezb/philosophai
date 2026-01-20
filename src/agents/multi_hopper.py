"""Multi-hop reasoning agent with community-routed GraphRAG."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

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
    ) -> dict:
        """
        Execute a multi-hop query with community-routed GraphRAG.

        Args:
            question: User's question
            max_hops: Maximum traversal depth
            max_context_chunks: Max chunks to send to LLM
            use_community_routing: If True, use community reports for routing

        Returns:
            Dict with answer, citations, and traversal trace
        """
        from .trace import TraceRecorder

        trace = TraceRecorder(query=question)

        # Step 1: Community report search (GraphRAG routing)
        self._emit("status", {"message": "Routing through community reports..."})
        
        target_communities = []
        community_cited_chunks = []
        community_scores = {}
        report_result = None
        
        if use_community_routing:
            try:
                report_result = self.vector_search.search_community_reports(question, limit=5)
                target_communities = report_result.community_ids
                community_cited_chunks = report_result.cited_chunk_ids
                community_scores = dict(zip(report_result.community_ids, report_result.scores))
                
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
        
        # Step 2: Vector search (optionally filtered to community chunks)
        self._emit("status", {"message": "Searching knowledge base..."})
        
        vector_result = self.vector_search.search(question, chunk_limit=15, community_limit=5)
        
        # Merge community-cited chunks with vector search results
        all_chunk_ids = list(dict.fromkeys(community_cited_chunks + vector_result.chunk_ids))
        
        trace.add_thought(
            f"Found {len(vector_result.chunk_ids)} chunks via vector search, {len(community_cited_chunks)} from community reports",
            action="vector_search",
            observation=f"Total unique chunks: {len(all_chunk_ids)}",
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
        
        traversal_trace = self.traverser.traverse(
            seed_nodes=seed_nodes,
            target_communities=traversal_communities,
            query=question,
            max_hops=max_hops,
            max_nodes=50,
            restrict_to_communities=bool(target_communities),  # Strict mode if we have community routing
        )
        traversal_trace.seed_chunks = all_chunk_ids[:20]

        trace.add_thought(
            f"Traversal complete: visited {len(traversal_trace.visited_nodes)} nodes, collected {len(traversal_trace.collected_chunk_ids)} chunks",
            action="graph_traverse",
            observation=f"Communities touched: {list(traversal_trace.visited_communities)}",
        )
        trace.communities_explored = list(traversal_trace.visited_communities)
        trace.nodes_visited = list(traversal_trace.visited_nodes)

        # Step 5: Select context (with must-keep guarantees)
        self._emit("status", {"message": "Selecting context..."})
        
        context_chunks, all_collected_ids, context_scores = self.fusion.get_context_with_selection(
            query=question,
            report_result=report_result if use_community_routing else None,
            vector_chunks=vector_result.chunk_ids,
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
