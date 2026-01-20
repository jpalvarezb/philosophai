"""Multi-hop reasoning agent with sequential thinking."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from openai import OpenAI
    from ..storage import DuckDBStorage
    from ..graph import GraphBuilder, GraphTraverser
    from ..rag import VectorSearch, ResultFusion, CitationBuilder
    from .tools import AgentTools
    from .trace import TraceRecorder


class MultiHopAgent:
    """
    Orchestrates multi-hop reasoning over the knowledge graph.
    
    Pipeline:
    1. Vector search (chunks + community summaries)
    2. Identify seed nodes and target communities
    3. Community-gated graph traversal
    4. Fuse results and select context
    5. Generate answer with citations
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
        llm_model: str = "gpt-4o",
        on_event: Callable[[dict], None] | None = None,  # For streaming updates
    ):
        self.storage = storage
        self.graph_builder = graph_builder
        self.vector_search = vector_search
        self.fusion = fusion
        self.citation_builder = citation_builder
        self.traverser = traverser
        self.llm_client = llm_client
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
    ) -> dict:
        """
        Execute a multi-hop query.
        
        Args:
            question: User's question
            max_hops: Maximum traversal depth
            max_context_chunks: Max chunks to send to LLM
        
        Returns:
            Dict with answer, citations, and traversal trace
        """
        from .trace import TraceRecorder

        trace = TraceRecorder(query=question)

        # Step 1: Vector search
        self._emit("status", {"message": "Searching knowledge base..."})
        trace.add_thought("Starting vector search for relevant chunks and communities")
        
        vector_result = self.vector_search.search(question, chunk_limit=10, community_limit=5)
        
        trace.add_thought(
            f"Found {len(vector_result.chunk_ids)} relevant chunks and {len(vector_result.community_ids)} relevant communities",
            action="vector_search",
            action_input={"query": question},
            observation=f"Chunks: {vector_result.chunk_ids[:5]}, Communities: {vector_result.community_ids}",
        )

        # Step 2: Get seed nodes from chunks
        self._emit("status", {"message": "Identifying seed entities..."})
        seed_nodes = self.storage.get_entity_ids_from_chunks(vector_result.chunk_ids)
        trace.add_thought(
            f"Identified {len(seed_nodes)} seed entities from chunks",
            observation=f"Seeds: {seed_nodes[:10]}",
        )
        trace.nodes_visited.extend(seed_nodes)

        # Step 3: Graph traversal
        self._emit("status", {"message": "Traversing knowledge graph..."})
        trace.add_thought(f"Beginning graph traversal from {len(seed_nodes)} seeds, targeting {len(vector_result.community_ids)} communities")
        
        traversal_trace = self.traverser.traverse(
            seed_nodes=seed_nodes,
            target_communities=vector_result.community_ids,
            max_hops=max_hops,
            max_nodes=50,
        )
        traversal_trace.query = question
        traversal_trace.seed_chunks = vector_result.chunk_ids

        trace.add_thought(
            f"Traversal complete: visited {len(traversal_trace.visited_nodes)} nodes, collected {len(traversal_trace.collected_chunk_ids)} chunks",
            action="graph_traverse",
            observation=f"Communities touched: {list(traversal_trace.visited_communities)}",
        )
        trace.communities_explored = list(traversal_trace.visited_communities)
        trace.nodes_visited.extend(list(traversal_trace.visited_nodes))

        # Step 4: Fuse results
        self._emit("status", {"message": "Ranking and selecting context..."})
        fused = self.fusion.fuse(vector_result, traversal_trace, max_chunks=30)
        context_chunks = self.fusion.get_context_chunks(fused, limit=max_context_chunks)
        
        trace.total_chunks_retrieved = len(fused.chunk_ids)
        trace.add_thought(
            f"Fused {len(fused.chunk_ids)} unique chunks, selected top {len(context_chunks)} for context",
        )

        # Step 5: Build citations
        chunk_ids_for_citations = [c[0] for c in context_chunks]
        citations = self.citation_builder.build_citations(chunk_ids_for_citations)

        # Step 6: Generate answer
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
